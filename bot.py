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
import logging

# Настройка подробного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('AlinaBot')

# Загружаем переменные из .env файла
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

# Счетчик сообщений для каждого канала (для автоответов)
channel_message_count = {}

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
        
        logger.info(f"💾 Сохранено сообщение от {author} в канал {channel_id}")
        return history

class AIAssistant:
    """Класс для работы с двумя AI моделями"""
    
    @staticmethod
    async def analyze_context(message_history, current_message, is_mentioned, message_count):
        """Gemma анализирует контекст и эмоции"""
        logger.info(f"🔍 Начало анализа контекста. Упоминание: {is_mentioned}, Счетчик: {message_count}")
        
        # Формируем контекст из последних сообщений
        context_text = "\n".join([
            f"{msg['author']}: {msg['content']}" 
            for msg in message_history[-10:]
        ])
        
        # Решаем отвечать ли на основе счетчика сообщений
        # Если упомянули - всегда отвечаем
        # Иначе отвечаем раз в 3-4 сообщения (случайно)
        auto_respond = is_mentioned or (message_count >= random.randint(3, 4))
        
        analysis_prompt = f"""Проанализируй следующий диалог в Discord чате:

{context_text}

Новое сообщение: {current_message}

Определи:
1. Основную тему разговора (одним словом)
2. Эмоциональная атмосфера (позитивная/нейтральная/негативная)
3. Стоит ли отвечать: {"да (упомянули бота)" if is_mentioned else ("да (можно вписаться)" if auto_respond else "нет (рано)")}
4. Если отвечать, то каким должен быть тон (дружелюбный/шутливый/поддерживающий/информативный)
5. Подходящий поисковый запрос для GIF (например: happy, laugh, thinking, love, excited, confused)

Ответь ТОЛЬКО в формате JSON:
{{"topic": "тема", "mood": "настроение", "should_respond": "да/нет", "tone": "тон", "gif_query": "запрос"}}"""
        
        try:
            logger.info("🤖 Отправка запроса к Gemma-2-27b...")
            response = gemini_client.models.generate_content(
                model="ggemma-3-27b-it",
                contents=analysis_prompt
            )
            
            # Парсим JSON из ответа
            response_text = response.text.strip()
            logger.info(f"📥 Получен ответ от Gemma: {response_text[:100]}...")
            
            # Ищем JSON в ответе
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                
                logger.info(f"✅ Анализ завершен: тема='{analysis['topic']}', настроение='{analysis['mood']}', ответить='{analysis['should_respond']}', тон='{analysis['tone']}'")
            else:
                # Fallback если не удалось распарсить
                analysis = {
                    "topic": "общение",
                    "mood": "нейтральная",
                    "should_respond": "да" if auto_respond else "нет",
                    "tone": "дружелюбный",
                    "gif_query": "smile"
                }
                logger.warning("⚠️ Не удалось распарсить JSON, используем fallback")
            
            return analysis
        except Exception as e:
            logger.error(f"❌ Ошибка анализа Gemma: {e}")
            return {
                "topic": "общение",
                "mood": "нейтральная",
                "should_respond": "да" if auto_respond else "нет",
                "tone": "дружелюбный",
                "gif_query": "smile"
            }
    
    @staticmethod
    async def generate_response(analysis, message_history, current_message, is_mentioned):
        """Mistral генерирует ответ на основе анализа"""
        logger.info(f"✍️ Начало генерации ответа. Развернутый: {is_mentioned}")
        
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
            logger.info("🤖 Отправка запроса к Mistral...")
            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=context_messages,
                max_tokens=150 if is_mentioned else 50,
                temperature=0.9
            )
            
            reply = response.choices[0].message.content.strip()
            # Убираем возможные префиксы имени бота
            reply = reply.replace("Ассистент:", "").replace("Бот:", "").strip()
            
            logger.info(f"✅ Ответ сгенерирован: '{reply}'")
            return reply
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Mistral: {e}")
            fallback = random.choice([
                "ахах точно 😄",
                "согласна!",
                "ну да)",
                "интересно 🤔",
                "ого"
            ])
            logger.info(f"⚠️ Используем fallback ответ: '{fallback}'")
            return fallback

class GifHelper:
    """Класс для работы с Giphy GIF API"""
    
    @staticmethod
    async def get_gif(query):
        """Получает случайный GIF по запросу через Giphy API"""
        if not GIPHY_API_KEY:
            logger.warning("⚠️ GIPHY_API_KEY не установлен")
            return None
        
        logger.info(f"🖼️ Поиск GIF по запросу: '{query}'")
            
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
                            gif_url = gif['images']['original']['url']
                            logger.info(f"✅ GIF найден: {gif_url}")
                            return gif_url
                        else:
                            logger.warning(f"⚠️ GIF не найден для запроса: '{query}'")
                    else:
                        logger.error(f"❌ Ошибка API Giphy: статус {response.status}")
        except Exception as e:
            logger.error(f"❌ Ошибка получения GIF: {e}")
        
        return None

@bot.event
async def on_ready():
    """Событие при запуске бота"""
    logger.info("=" * 60)
    logger.info(f"🚀 Бот {bot.user} успешно запущен!")
    logger.info(f"🆔 ID: {bot.user.id}")
    logger.info(f"📊 Подключен к {len(bot.guilds)} серверам")
    logger.info("=" * 60)

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
    
    logger.info("=" * 60)
    logger.info(f"📨 Новое сообщение в канале {message.channel.name} ({channel_id})")
    logger.info(f"👤 Автор: {message.author.name}")
    logger.info(f"💬 Содержание: {message.content}")
    
    # Сохраняем сообщение в лог
    history = ChatLogger.save_message(
        channel_id,
        message.author.name,
        message.content,
        datetime.now().isoformat()
    )
    
    # Увеличиваем счетчик сообщений для канала
    if channel_id not in channel_message_count:
        channel_message_count[channel_id] = 0
    channel_message_count[channel_id] += 1
    
    # Проверяем упоминание бота
    is_mentioned = bot.user.mentioned_in(message)
    
    if is_mentioned:
        logger.info("🔔 Бот упомянут в сообщении!")
    
    # Анализируем контекст с помощью Gemma
    analysis = await AIAssistant.analyze_context(
        history,
        message.content,
        is_mentioned,
        channel_message_count[channel_id]
    )
    
    # Решаем, отвечать ли
    should_respond = analysis['should_respond'].lower() == 'да'
    
    if not should_respond and not is_mentioned:
        logger.info(f"⏭️ Пропускаем ответ. Счетчик сообщений: {channel_message_count[channel_id]}")
        logger.info("=" * 60)
        return
    
    # Сбрасываем счетчик после ответа
    channel_message_count[channel_id] = 0
    logger.info("🔄 Счетчик сообщений сброшен")
    
    # Показываем "печатает..."
    logger.info("⌨️ Показываем статус 'печатает...'")
    async with message.channel.typing():
        # Генерируем ответ с помощью Mistral
        response_text = await AIAssistant.generate_response(
            analysis,
            history,
            message.content,
            is_mentioned
        )
        
        # Отправляем текстовый ответ
        logger.info(f"📤 Отправка текстового ответа: '{response_text}'")
        await message.channel.send(response_text)
        logger.info("✅ Текстовый ответ отправлен")
        
        # 70% шанс отправить GIF (чтобы не спамить)
        gif_chance = random.random()
        logger.info(f"🎲 Шанс GIF: {gif_chance:.2f} (порог: 0.70)")
        
        if gif_chance < 0.3:
            gif_url = await GifHelper.get_gif(analysis['gif_query'])
            if gif_url:
                logger.info(f"📤 Отправка GIF: {gif_url}")
                await message.channel.send(gif_url)
                logger.info("✅ GIF отправлен")
            else:
                logger.info("⏭️ GIF не найден, пропускаем")
        else:
            logger.info("⏭️ GIF не отправляем (не прошли шанс)")
    
    logger.info("=" * 60)

@bot.command(name='clear_history')
@commands.has_permissions(administrator=True)
async def clear_history(ctx):
    """Команда для очистки истории чата (только для администраторов)"""
    channel_id = str(ctx.channel.id)
    filename = ChatLogger.get_log_filename(channel_id)
    
    logger.info(f"🗑️ Команда очистки истории в канале {channel_id}")
    
    if os.path.exists(filename):
        os.remove(filename)
        logger.info(f"✅ История канала {channel_id} очищена")
        await ctx.send("✅ История чата очищена!")
    else:
        logger.info(f"⚠️ История канала {channel_id} уже пуста")
        await ctx.send("История чата уже пуста.")

@bot.command(name='ping')
async def ping(ctx):
    """Проверка работы бота"""
    latency = round(bot.latency * 1000)
    logger.info(f"🏓 Команда ping. Задержка: {latency}мс")
    await ctx.send(f'Понг! 🏓 Задержка: {latency}мс')

# Запуск бота
if __name__ == "__main__":
    logger.info("🔧 Запуск бота...")
    bot.run(DISCORD_TOKEN)
