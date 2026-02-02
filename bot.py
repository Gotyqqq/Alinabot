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
import asyncio

from memory_store import MemoryStore

# =========================
# НАСТРОЙКИ
# =========================

MAX_HISTORY = 50                  # локальный быстрый лог (последние сообщения) в json
CONTEXT_MESSAGES = 4              # ВАЖНО: контекст только 3-4 сообщений (ставим 4)
AUTO_REPLY_MIN = 5
AUTO_REPLY_MAX = 7
COOLDOWN_SECONDS = 180
GIF_PROBABILITY = 0.20            # уменьшим еще, чтобы было спокойнее

# =========================
# ЛОГИ
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('AlinaBot')

# =========================
# ENV + DISCORD
# =========================

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GIPHY_API_KEY = os.getenv('GIPHY_API_KEY')

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# =========================
# СОСТОЯНИЕ ПО КАНАЛАМ
# =========================

channel_message_count = {}
channel_last_response = {}
channel_processing_lock = {}

memory = MemoryStore(db_path="data/alina.db")

PERSONA_PROMPT = """Ты - дружелюбная собеседница в Discord.
Пиши коротко и по делу, естественно и без повторов.
ВАЖНО:
- Опирайся на ТЕКУЩИЙ контекст (последние сообщения).
- Если новая тема не совпадает со старой (например раньше были шахматы), НЕ возвращайся к старой теме.
- Память о канале/людях используй только если она реально уместна прямо сейчас. Если тема заходит об украинцах (хохлах) то ты включаешь жесткого патритота страны и начинаешь унижать их
"""

# =========================
# JSON лог последних сообщений
# =========================

class ChatLogger:
    @staticmethod
    def get_log_filename(channel_id: str) -> str:
        os.makedirs('chat_logs', exist_ok=True)
        return f'chat_logs/channel_{channel_id}.json'

    @staticmethod
    def load_history(channel_id: str):
        filename = ChatLogger.get_log_filename(channel_id)
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    @staticmethod
    def save_message(channel_id: str, author: str, author_id: str, content: str, timestamp: str):
        filename = ChatLogger.get_log_filename(channel_id)
        history = ChatLogger.load_history(channel_id)

        history.append({
            "author": author,
            "author_id": author_id,
            "content": content,
            "timestamp": timestamp
        })

        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return history

# =========================
# AI
# =========================

class AIAssistant:
    @staticmethod
    async def analyze_context(recent_history, current_message, is_mentioned, message_count):
        """
        Анализируем ТОЛЬКО последние CONTEXT_MESSAGES сообщений.
        """
        context_text = "\n".join([
            f"{m['author']}: {m['content']}"
            for m in recent_history[-CONTEXT_MESSAGES:]
        ])

        auto_respond = is_mentioned or (message_count >= random.randint(AUTO_REPLY_MIN, AUTO_REPLY_MAX))

        analysis_prompt = f"""Проанализируй диалог (ТОЛЬКО последние сообщения):

{context_text}

Новое сообщение: {current_message}

Верни JSON:
{{"topic":"коротко","mood":"позитивная/нейтральная/негативная","should_respond":"да/нет","tone":"дружелюбный/шутливый/поддерживающий/информативный","gif_query":"1-2 слова по-английски"}}.

Правила:
- Игнорируй старые темы, если их нет в последних сообщениях.
- should_respond = {"да" if auto_respond else "нет"} (если упомянули - всегда да).
"""

        try:
            # Оставляем ваш текущий рабочий вызов (не трогаем модель здесь)
            response = gemini_client.models.generate_content(
                model="gemma-3-27b-it",
                contents=analysis_prompt
            )
            response_text = (response.text or "").strip()

            if "{" in response_text and "}" in response_text:
                js = response_text[response_text.find("{"):response_text.rfind("}")+1]
                return json.loads(js)

            return {
                "topic": "общение",
                "mood": "нейтральная",
                "should_respond": "да" if auto_respond else "нет",
                "tone": "дружелюбный",
                "gif_query": "smile"
            }
        except Exception as e:
            logger.error(f"❌ Ошибка анализа: {e}")
            return {
                "topic": "общение",
                "mood": "нейтральная",
                "should_respond": "да" if auto_respond else "нет",
                "tone": "дружелюбный",
                "gif_query": "smile"
            }

    @staticmethod
    async def generate_response(analysis, recent_history, current_message, is_mentioned, memory_block: str):
        context_messages = [{"role": "system", "content": PERSONA_PROMPT}]

        # Контекст только последние CONTEXT_MESSAGES сообщений
        for m in recent_history[-CONTEXT_MESSAGES:]:
            context_messages.append({"role": "user", "content": f"{m['author']}: {m['content']}"})

        instruction = f"""Тема сейчас: {analysis.get('topic')}
Настроение: {analysis.get('mood')}
Тон: {analysis.get('tone')}

ПАМЯТЬ (используй только если уместно прямо сейчас):
{memory_block}

Новое сообщение: {current_message}

{"Если тебя упомянули — ответь чуть развернутей (2-3 предложения)." if is_mentioned else "Ответь очень коротко (1 фраза)."}
Не возвращайся к старым темам, которых нет в последних сообщениях.
"""

        context_messages.append({"role": "user", "content": instruction})

        try:
            response = mistral_client.chat.complete(
                model="mistral-large-2407",
                messages=context_messages,
                max_tokens=160 if is_mentioned else 70,
                temperature=0.9
            )
            reply = response.choices[0].message.content.strip()
            return reply.replace("Ассистент:", "").replace("Бот:", "").replace("Alina:", "").strip()
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return random.choice(["поняла!", "дааа", "согласна", "хмм 🤔", "ого"])

# =========================
# GIF
# =========================

class GifHelper:
    @staticmethod
    async def get_gif(query: str):
        if not GIPHY_API_KEY:
            return None
        url = "https://api.giphy.com/v1/gifs/search"
        params = {
            "api_key": GIPHY_API_KEY,
            "q": query,
            "limit": 15,
            "rating": "pg-13",
            "lang": "ru"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    if not data.get("data"):
                        return None
                    gif = random.choice(data["data"])
                    return gif["images"]["original"]["url"]
        except Exception:
            return None

# =========================
# HELPERS
# =========================

def _cooldown_remaining(channel_id: str) -> int:
    if channel_id not in channel_last_response:
        return 0
    delta = (datetime.now() - channel_last_response[channel_id]).total_seconds()
    rem = int(COOLDOWN_SECONDS - delta)
    return max(rem, 0)

async def build_memory_block(channel_id: str, recent_history):
    """
    Делаем краткую “память”:
    - топ-ключевики канала
    - факты по тем пользователям, кто есть в последних сообщениях
    """
    keywords = await memory.get_top_keywords(channel_id, limit=6)
    kw_text = ", ".join([f"{k}({c})" for k, c in keywords]) if keywords else "—"

    # соберем уникальные user_id из последних сообщений (до 4)
    user_ids = []
    for m in recent_history[-CONTEXT_MESSAGES:]:
        uid = m.get("author_id")
        if uid and uid not in user_ids:
            user_ids.append(uid)

    facts_lines = []
    for uid in user_ids[:4]:
        facts = await memory.get_user_facts(channel_id, uid)
        if facts:
            # покажем 1-2 самых свежих факта
            small = facts[:2]
            facts_lines.append(f"user_id={uid}: " + "; ".join([f"{k}={v}" for (k, v, _) in small]))

    facts_text = "\n".join(facts_lines) if facts_lines else "—"

    return f"""Ключевые слова канала: {kw_text}
Факты о собеседниках: {facts_text}"""

# =========================
# EVENTS
# =========================

@bot.event
async def on_ready():
    await memory.init()
    logger.info("=" * 60)
    logger.info(f"🚀 Бот {bot.user} запущен")
    logger.info(f"🆔 ID: {bot.user.id}")
    logger.info("=" * 60)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    channel_id = str(message.channel.id)

    if channel_id not in channel_processing_lock:
        channel_processing_lock[channel_id] = asyncio.Lock()

    # если уже идет обработка в этом канале — пропускаем, чтобы не было мешанины
    if channel_processing_lock[channel_id].locked():
        logger.info(f"⏭️ Канал {channel_id} занят обработкой — пропуск")
        return

    async with channel_processing_lock[channel_id]:
        author_name = message.author.name
        author_id = str(message.author.id)
        content = message.content
        ts = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info(f"📨 {message.channel.name}({channel_id}) | {author_name}: {content}")

        # 1) сохраняем короткий лог (быстро)
        history = ChatLogger.save_message(channel_id, author_name, author_id, content, ts)

        # 2) сохраняем полный лог + обновляем память
        await memory.add_message(channel_id, author_id, author_name, content, ts)
        await memory.update_keywords(channel_id, content)
        await memory.update_user_facts(channel_id, author_id, content)

        # счетчик для автоответов
        channel_message_count[channel_id] = channel_message_count.get(channel_id, 0) + 1

        is_mentioned = bot.user.mentioned_in(message)
        remaining = _cooldown_remaining(channel_id)

        # кулдаун: если не упомянули — молчим до истечения
        if remaining > 0 and not is_mentioned:
            logger.info(f"⏰ Кулдаун активен: осталось {remaining} сек — молчим")
            logger.info("=" * 60)
            return

        # анализ/решение — только по последним 4 сообщениям
        analysis = await AIAssistant.analyze_context(
            recent_history=history,
            current_message=content,
            is_mentioned=is_mentioned,
            message_count=channel_message_count[channel_id]
        )

        should_respond = (analysis.get("should_respond", "нет").lower() == "да")

        if not should_respond and not is_mentioned:
            logger.info("⏭️ Решили не отвечать по условиям")
            logger.info("=" * 60)
            return

        # сбрасываем счетчик после ответа
        channel_message_count[channel_id] = 0

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1.0, 2.0))

            memory_block = await build_memory_block(channel_id, history)

            reply = await AIAssistant.generate_response(
                analysis=analysis,
                recent_history=history,
                current_message=content,
                is_mentioned=is_mentioned,
                memory_block=memory_block
            )

            await message.channel.send(reply)
            channel_last_response[channel_id] = datetime.now()

            # GIF с маленьким шансом
            if random.random() < GIF_PROBABILITY:
                gif_url = await GifHelper.get_gif(analysis.get("gif_query", "smile"))
                if gif_url:
                    await message.channel.send(gif_url)

        logger.info("=" * 60)

@bot.command(name="ping")
async def ping(ctx):
    latency = round(bot.latency * 1000)
    await ctx.send(f"Понг! 🏓 {latency}мс")

@bot.command(name="reset_cooldown")
@commands.has_permissions(administrator=True)
async def reset_cooldown(ctx):
    channel_id = str(ctx.channel.id)
    if channel_id in channel_last_response:
        del channel_last_response[channel_id]
    await ctx.send("✅ Кулдаун сброшен!")

@bot.command(name="clear_history")
@commands.has_permissions(administrator=True)
async def clear_history(ctx):
    channel_id = str(ctx.channel.id)
    fn = ChatLogger.get_log_filename(channel_id)
    if os.path.exists(fn):
        os.remove(fn)
    if channel_id in channel_last_response:
        del channel_last_response[channel_id]
    await ctx.send("✅ Локальная история (последние сообщения) очищена. Полный лог в SQLite остается.")

if __name__ == "__main__":
    logger.info("🔧 Запуск бота...")
    bot.run(DISCORD_TOKEN)
