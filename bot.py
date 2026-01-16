import discord
from discord.ext import commands
import os
import json
import random
import aiohttp
from datetime import datetime
from google import genai
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

# Настройка intents для Discord
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)

# API ключи
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GIPHY_API_KEY = os.getenv('GIPHY_API_KEY')

# Инициализация AI клиентов
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Хранилище истории для каждого канала
channel_histories = {}
MAX_HISTORY = 50

# Системный промпт для персоны молодой девушки
PERSONA_PROMPT = """Ты - молодая девушка, общаешься в Discord чате со своими друзьями. 
Ты естественна, дружелюбна, используешь современный сленг и эмодзи. 
Никогда не выдавай что ты бот или AI. Ты настоящий человек с эмоциями и мнениями.
Пиши коротко и по теме, как обычная девушка в чате. Используй смайлики, но не перебарщивай.
Отвечай на русском языке естественно и непринужденно."""

class ChatLogger:
    """Класс для управления логами чатов"""
    
    @staticmethod
    def get_log_filename(channel_id):
        """Создает имя файла для конкретного канала"""
        os.makedirs('chat_logs', exist_ok=True)
        return f'chat_logs/channel_{channel_id}.json'
    
    @staticmethod
    def load_history(channel_id):
        """Загружает историю чата из файла"""
        filename = ChatLogger.get_log_filename(channel_id)
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    @staticmethod
    def save_message(channel_id, author, content, timestamp):
        """Сохраняет сообщение в лог"""
        filename = ChatLogger.get_log_filename(channel_id)
        history = ChatLogger.load_history(channel_id)
        
        history.append({
            'author': author,
            'content': content,
            'timestamp': timestamp
        })
        
        # Ограничиваем размер истории
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return history

class AIAssistant:
    """Класс для работы с двумя AI моделями"""
    
    @staticmethod
    async def analyze_context(message_history, current_message, is_mentioned):
        """Gemma анализирует контекст и эмоции"""
        # Формируем контекст из последних сообщений
        context_text = "\n".join([
            f"{msg['author']}: {msg['content']}" 
            for msg in message_history[-10:]
        ])
        
        analysis_prompt = f"""Проанализируй следующий диалог в Discord чате:

{context_text}

Новое сообщение: {current_message}

Определи:
1. Основную тему разговора (одним словом)
2. Эмоциональная атмосфера (позитивная/нейтральная/негативная)
3. Стоит ли отвечать (да/нет) - отвечай только если:
   - Упомянули бота: {"да" if is_mentioned else "нет"}
   - Или если можно естественно вписаться в диалог (но не чаще чем раз в 5-7 сообщений)
4. Если отвечать, то каким должен быть тон (дружелюбный/шутливый/поддерживающий/информативный)
5. Подходящий поисковый запрос для GIF (например: happy, laugh, thinking, love, excited, confused)

Ответь ТОЛЬКО в формате JSON:
{{"topic": "тема", "mood": "настроение", "should_respond": "да/нет", "tone": "тон", "gif_query": "запрос"}}"""
        
        try:
            response = gemini_client.models.generate_content(
                model="gemma-3-27b",
                contents=analysis_prompt
            )
            
            # Парсим JSON из ответа
            response_text = response.text.strip()
            # Ищем JSON в ответе
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
            else:
                # Fallback если не удалось распарсить
                analysis = {
                    "topic": "общение",
                    "mood": "нейтральная",
                    "should_respond": "да" if is_mentioned else "нет",
                    "tone": "дружелюбный",
                    "gif_query": "smile"
                }
            
            return analysis
        except Exception as e:
            print(f"Ошибка анализа Gemma: {e}")
            return {
                "topic": "общение",
                "mood": "нейтральная",
                "should_respond": "да" if is_mentioned else "нет",
                "tone": "дружелюбный",
                "gif_query": "smile"
            }
    
    @staticmethod
    async def generate_response(analysis, message_history, current_message, is_mentioned):
        """Mistral генерирует ответ на основе анализа"""
        # Формируем контекст
        context_messages = []
        
        # Добавляем системный промпт
        context_messages.append({
            "role": "system",
            "content": PERSONA_PROMPT
        })
        
        # Добавляем последние сообщения как контекст
        for msg in message_history[-10:]:
            context_messages.append({
                "role": "user",
                "content": f"{msg['author']}: {msg['content']}"
            })
        
        # Инструкция для генерации ответа
        instruction = f"""Текущая тема: {analysis['topic']}
Настроение: {analysis['mood']}
Тон ответа: {analysis['tone']}

Новое сообщение: {current_message}

{"Тебя упомянули - ответь более развернуто (2-3 предложения)." if is_mentioned else "Ответь коротко и естественно (1 короткое предложение или фраза)."}

Помни: ты молодая девушка, общаешься с друзьями. Будь естественной, не выдавай что ты бот."""
        
        context_messages.append({
            "role": "user",
            "content": instruction
        })
        
        try:
            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=context_messages,
                max_tokens=150 if is_mentioned else 50,
                temperature=0.9
            )
            
            reply = response.choices[0].message.content.strip()
            # Убираем возможные префиксы имени бота
            reply = reply.replace("Ассистент:", "").replace("Бот:", "").strip()
            
            return reply
        except Exception as e:
            print(f"Ошибка генерации Mistral: {e}")
            return random.choice([
                "ахах точно 😄",
                "согласна!",
                "ну да)",
                "интересно 🤔",
                "ого"
            ])

class GifHelper:
    """Класс для работы с Giphy GIF API"""
    
    @staticmethod
    async def get_gif(query):
        """Получает случайный GIF по запросу через Giphy API"""
        if not GIPHY_API_KEY:
            return None
            
        url = "https://api.giphy.com/v1/gifs/search"
        params = {
            'api_key': GIPHY_API_KEY,
            'q': query,
            'limit': 20,
            'rating': 'pg-13',
            'lang': 'ru'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            gif = random.choice(data['data'])
                            # Возвращаем URL GIF
                            return gif['images']['original']['url']
        except Exception as e:
            print(f"Ошибка получения GIF: {e}")
        
        return None

@bot.event
async def on_ready():
    """Событие при запуске бота"""
    print(f'Бот {bot.user} успешно запущен!')
    print(f'ID: {bot.user.id}')
    print('------')

@bot.event
async def on_message(message):
    """Обработка входящих сообщений"""
    # Игнорируем сообщения от самого бота
    if message.author == bot.user:
        return
    
    # Игнорируем команды
    if message.content.startswith('!'):
        await bot.process_commands(message)
        return
    
    channel_id = str(message.channel.id)
    
    # Сохраняем сообщение в лог
    history = ChatLogger.save_message(
        channel_id,
        message.author.name,
        message.content,
        datetime.now().isoformat()
    )
    
    # Проверяем упоминание бота
    is_mentioned = bot.user.mentioned_in(message)
    
    # Анализируем контекст с помощью Gemma
    analysis = await AIAssistant.analyze_context(
        history,
        message.content,
        is_mentioned
    )
    
    # Решаем, отвечать ли
    should_respond = analysis['should_respond'].lower() == 'да'
    
    if not should_respond and not is_mentioned:
        return
    
    # Показываем "печатает..."
    async with message.channel.typing():
        # Генерируем ответ с помощью Mistral
        response_text = await AIAssistant.generate_response(
            analysis,
            history,
            message.content,
            is_mentioned
        )
        
        # Отправляем текстовый ответw
        await message.channel.send(response_text)
        
        # 70% шанс отправить GIF (чтобы не спамить)
        if random.random() < 0.7:
            gif_url = await GifHelper.get_gif(analysis['gif_query'])
            if gif_url:
                await message.channel.send(gif_url)

@bot.command(name='clear_history')
@commands.has_permissions(administrator=True)
async def clear_history(ctx):
    """Команда для очистки истории чата (только для администраторов)"""
    channel_id = str(ctx.channel.id)
    filename = ChatLogger.get_log_filename(channel_id)
    if os.path.exists(filename):
        os.remove(filename)
        await ctx.send("✅ История чата очищена!")
    else:
        await ctx.send("История чата уже пуста.")

@bot.command(name='ping')
async def ping(ctx):
    """Проверка работы бота"""
    await ctx.send(f'Понг! 🏓 Задержка: {round(bot.latency * 1000)}мс')

# Запуск бота
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
