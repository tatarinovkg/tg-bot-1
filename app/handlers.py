import sqlite3
import time
import re
from aiogram import Router, types, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from config import DB_PATH, GROUP_ID, ADMIN_IDS, bot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = Router()

rules_text = (
    "Бот предназначен для контроля соблюдения правил участниками группы 🌲 КП Зелёные Холмы 🌻.\n"
    "Общие правила группы размещены в закрепленном сообщении темы General. "
    "С правилом каждой отдельной темы в канале можно также ознакомиться в закрепленном сообщении внутри тем.\n\n"
    "*Список действий, которые расцениваются как нарушения правил:*\n" 
    "1)  Одно и то же объявление можно размещать не чаще, чем раз в 5 дней!\n"
    "2)  Дублирование одного и того же сообщения в разных темах канала запрещается!\n"
    "3)  Размещение коммерческой рекламы в темах, не предназначенных для этого, запрещается! "
    "(см. закрепленные сообщения в каждой теме)\n"
    "4)  Оскорбления, угрозы, грубость противоречат правилам канала!\n\n"
    "Бот отправляет сообщение предупреждение, и по достижению 3 предупреждений пользователь блокируется на 5 дней "
    "(не сможет отправлять сообщения в канал). Также предусмотрен ручной контроль соблюдения правил со стороны "
    "администраторов канала.\n\n"
    "Если вы увидели ошибку в размещенном объявлении или просто в вашем сообщении, "
    "*то рекомендуем вам пользоваться режимом редактирования (!)*, " 
    "а не удалять сообщение, чтобы разместить его заново: в течение 5 дней бот не даст вам отправить похожее сообщение в группу. " 
    "В таких случаях рекомендуем обратиться к администратору бота @tatarinovkg."
)

@router.message(CommandStart())
async def start(message: Message):
    await message.answer(rules_text, parse_mode="Markdown")

# Нормализация текста
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def get_ad_record(user_id: int, photo_id: str, text: str, thread_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    topic_settings = get_topic_settings(thread_id)
    ad_frequency_seconds = topic_settings["ad_frequency_days"] * 24 * 60 * 60
    current_time = int(time.time())
    time_threshold = current_time - ad_frequency_seconds

    result = None

    # Если есть фото, ищем по photo_id
    if photo_id:
        cursor.execute(
            "SELECT id, thread_id, timestamp, text FROM ads WHERE user_id=? AND photo_id=? AND timestamp >= ?",
            (user_id, photo_id, time_threshold)
        )
        result = cursor.fetchone()  # (id, thread_id, timestamp, text) или None

    # Если есть текст, ищем по text (если фото не найдено или его нет)
    if text and not result:
        cursor.execute(
            "SELECT id, thread_id, timestamp, text FROM ads WHERE user_id=? AND text=? AND timestamp >= ?",
            (user_id, text, time_threshold)
        )
        result = cursor.fetchone()  # (id, thread_id, timestamp, text) или None

    conn.close()
    return result

# Вставка новой записи в БД
def insert_ad_record(user_id: int, thread_id: int, text: str, photo_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = int(time.time())
    cursor.execute(
        "INSERT INTO ads (user_id, thread_id, text, photo_id, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, thread_id, text, photo_id, current_time)
    )
    conn.commit()
    conn.close()

# Функция обновления записи рекламы
def update_ad_record(record_id: int, new_thread_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = int(time.time())
    cursor.execute("UPDATE ads SET timestamp=?, thread_id=? WHERE id=?", (current_time, new_thread_id, record_id))
    conn.commit()
    conn.close()

def get_ad_warnings(user_id: int, ad_key: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT warning_count FROM warnings WHERE user_id=? AND ad_key=?", (user_id, ad_key))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0

def increase_ad_warnings(user_id: int, ad_key: str):
    current_warnings = get_ad_warnings(user_id, ad_key)
    conn = get_db_connection()
    cursor = conn.cursor()
    if current_warnings == 0:
        cursor.execute(
            "INSERT INTO warnings (user_id, ad_key, warning_count, last_warning) VALUES (?, ?, ?, ?)",
            (user_id, ad_key, 1, int(time.time()))
        )
    else:
        cursor.execute(
            "UPDATE warnings SET warning_count = warning_count + 1, last_warning=? WHERE user_id=? AND ad_key=?",
            (int(time.time()), user_id, ad_key)
        )
    conn.commit()
    conn.close()
    return current_warnings + 1

def reset_ad_warnings(user_id: int, ad_key: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM warnings WHERE user_id=? AND ad_key=?", (user_id, ad_key))
    conn.commit()
    conn.close()

def get_topic_settings(thread_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT enabled, block_days, warnings_limit, ad_frequency_days FROM topics WHERE thread_id=?", (thread_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            "enabled": bool(result[0]),
            "block_days": result[1],
            "warnings_limit": result[2],
            "ad_frequency_days": result[3]
        }
    else:
        return {
            "enabled": True,
            "block_days": 5,
            "warnings_limit": 3,
            "ad_frequency_days": 5  # Значение по умолчанию
        }

def ensure_topic_exists(thread_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM topics WHERE thread_id=?", (thread_id,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO topics (thread_id, enabled, block_days, warnings_limit) VALUES (?, ?, ?, ?)",
            (thread_id, 1, 5, 3)
        )
    conn.commit()
    conn.close()

# Если message.message_thread_id отсутствует, считаем, что это тема General (thread_id = 0)
def get_thread_id(message: Message) -> int:
    return message.message_thread_id if message.message_thread_id is not None else 0

# Отправка уведомления в тему General (thread_id = 0)
async def notify_general(text: str):
    try:
        await bot.send_message(chat_id=GROUP_ID, message_thread_id=0, text=text, parse_mode="HTML")
    except Exception as e:
        print("Ошибка отправки уведомления в General:", e)

async def notify_admins_about_ban(user_id: int, first_name: str, reason: str):
    user_link = f'<a href="tg://user?id={user_id}">{first_name}</a>'
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=int(admin_id),
                text=f"Пользователь {user_link} (ID: {user_id}) заблокирован по причине: {reason}",
                parse_mode="HTML"
            )
        except Exception as e:
            print(f"Не удалось отправить сообщение администратору {admin_id}: {e}")

async def notify_admins_suspicious_similarity(user_id: int, first_name: str, current_text: str, previous_text: str, similarity: float, current_message_link: str):
    user_link = f'<a href="tg://user?id={user_id}">{first_name}</a>'
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(
                chat_id=int(admin_id),
                text=(
                    f"Обнаружено подозрительное сообщение от {user_link} (ID: {user_id}).\n"
                    f"Текущий текст: <code>{current_text}</code>\n"
                    f"Предыдущий текст: <code>{previous_text}</code>\n"
                    f"Схожесть: {similarity:.2%}\n"
                    f"Ссылка на текущее сообщение: {current_message_link}"
                ),
                parse_mode="HTML"
            )
        except Exception as e:
            print(f"Не удалось уведомить админа {admin_id}: {e}")


@router.message(F.chat.id == GROUP_ID)
async def handle_group_message(message: types.Message):
    thread_id = get_thread_id(message)
    ensure_topic_exists(thread_id)
    topic_settings = get_topic_settings(thread_id)
    if not topic_settings["enabled"]:
        return

    user_id = message.from_user.id
    first_name = message.from_user.first_name
    user_link = f'<a href="tg://user?id={user_id}">{first_name}</a>'
    ad_frequency_seconds = topic_settings["ad_frequency_days"] * 24 * 60 * 60

    # Определяем текст и фото
    text_content = message.text or message.caption or ""  # Текст или подпись
    norm_text = normalize_text(text_content) if text_content else ""
    photo_id = message.photo[-1].file_id if message.photo else ""  # ID последнего фото

    # Игнорируем короткие сообщения без фото
    if not photo_id and len(text_content) < 20:
        return

    current_time = int(time.time())
    violation = False
    violation_reason = ""
    matched_ad_key = norm_text if norm_text else photo_id

    # Получаем все предыдущие записи пользователя за последние 5 дней
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text, photo_id, timestamp, thread_id FROM ads WHERE user_id=? AND timestamp >= ?",
        (user_id, current_time - ad_frequency_seconds)
    )
    previous_ads = cursor.fetchall()  # (text, photo_id, timestamp, thread_id)
    conn.close()

    # Проверка текста (если он есть)
    if norm_text:
        ad_record = get_ad_record(user_id, "", norm_text, thread_id)
        if ad_record:
            violation = True
            matched_ad_key = norm_text
            date_str = time.strftime('%d.%m.%Y в %H:%M', time.localtime(ad_record[2]))
            if ad_record[1] == thread_id:
                violation_reason = f"Вы уже размещали это объявление в этой теме {date_str}."
            else:
                violation_reason = f"Вы уже размещали это объявление в другой теме {date_str}."
        else:
            for prev_text, prev_photo_id, prev_timestamp, prev_thread_id in previous_ads:
                if prev_text and not prev_photo_id:
                    documents = [prev_text, norm_text]
                    vectorizer = TfidfVectorizer()
                    try:
                        matrix = vectorizer.fit_transform(documents)
                        similarity = cosine_similarity(matrix)[0][1]
                        if similarity >= 0.75:
                            violation = True
                            matched_ad_key = prev_text
                            date_str = time.strftime('%d.%m.%Y в %H:%M', time.localtime(prev_timestamp))
                            similarity_percent = int(similarity * 100)
                            if prev_thread_id == thread_id:
                                violation_reason = (
                                    f"Ваше сообщение слишком похоже (схожесть {similarity_percent}%) "
                                    f"на объявление, которое вы разместили в этой теме {date_str}."
                                )
                            else:
                                violation_reason = (
                                    f"Ваше сообщение слишком похоже (схожесть {similarity_percent}%) "
                                    f"на объявление, которое вы разместили в другой теме {date_str}."
                                )
                            break
                        elif 0.35 <= similarity < 0.75:
                            current_message_link = f"https://t.me/c/{str(GROUP_ID)[4:]}/{message.message_id}"
                            await notify_admins_suspicious_similarity(
                                user_id, first_name, text_content, prev_text, similarity, current_message_link
                            )
                    except ValueError as e:
                        print(f"Ошибка при вычислении схожести: {e}")

    # Проверка фото (если оно есть)
    if photo_id and not violation:
        ad_record = get_ad_record(user_id, photo_id, "", thread_id)
        if ad_record:
            violation = True
            matched_ad_key = photo_id
            date_str = time.strftime('%d.%m.%Y в %H:%M', time.localtime(ad_record[2]))
            if ad_record[1] == thread_id:
                violation_reason = f"Вы уже размещали это фото в этой теме {date_str}."
            else:
                violation_reason = f"Вы уже размещали это фото в другой теме {date_str}."
        elif norm_text and ad_record and ad_record[3]:
            documents = [ad_record[3], norm_text]
            vectorizer = TfidfVectorizer()
            try:
                matrix = vectorizer.fit_transform(documents)
                similarity = cosine_similarity(matrix)[0][1]
                if similarity >= 0.75:
                    violation = True
                    matched_ad_key = ad_record[3]
                    date_str = time.strftime('%d.%m.%Y в %H:%M', time.localtime(ad_record[2]))
                    similarity_percent = int(similarity * 100)
                    if ad_record[1] == thread_id:
                        violation_reason = (
                            f"Ваше фото с похожим текстом (схожесть {similarity_percent}%) "
                            f"уже было размещено в этой теме {date_str}."
                        )
                    else:
                        violation_reason = (
                            f"Ваше фото с похожим текстом (схожесть {similarity_percent}%) "
                            f"уже было размещено в другой теме {date_str}."
                        )
            except ValueError as e:
                print(f"Ошибка при вычислении схожести: {e}")

    # Обработка нарушения
    if violation:
        warning_count = increase_ad_warnings(user_id, matched_ad_key)
        if warning_count >= topic_settings["warnings_limit"]:
            block_seconds = topic_settings["block_days"] * 24 * 3600 if topic_settings["block_days"] > 0 else 0
            banned_until = current_time + block_seconds if block_seconds > 0 else 0
            try:
                await bot.restrict_chat_member(
                    chat_id=GROUP_ID,
                    user_id=user_id,
                    permissions=types.ChatPermissions(can_send_messages=False),
                    until_date=banned_until
                )
                block_duration = 'навсегда' if topic_settings['block_days'] == 0 else f'на {topic_settings["block_days"]} дней'
                block_message = (
                    f"🚫 {user_link}, {violation_reason}\n"
                    f"Вы были заблокированы {block_duration} за повторные нарушения.\n"
                    f"Ознакомьтесь с правилами: <a href=\"https://t.me/greenHillsRulesBot?start=start\">Правила</a>."
                )
                await message.answer(block_message, disable_web_page_preview=True, parse_mode="HTML")
                await notify_admins_about_ban(user_id, first_name, "Повторные нарушения")
                add_ban(user_id, first_name, banned_until, "Повторные нарушения")
            except Exception as e:
                print(f"Ошибка при блокировке: {e}")
            reset_ad_warnings(user_id, matched_ad_key)
        else:
            warning_message = (
                f"⚠️ {user_link}, ваше сообщение удалено: {violation_reason}\n"
                f"Предупреждение № {warning_count}/{topic_settings['warnings_limit']}.\n"
                f"Ознакомьтесь с <a href=\"https://t.me/greenHillsRulesBot?start=start\">правилами</a>."
            )
            await message.reply(warning_message, disable_web_page_preview=True, parse_mode="HTML")
        await message.delete()
    else:
        insert_ad_record(user_id, thread_id, norm_text, photo_id)

@router.message(F.chat.id == GROUP_ID)
async def handle_suspicious(message: types.Message):
    pass

@router.message(Command("ban"))
async def admin_ban(message: types.Message):
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in ADMIN_IDS:
        return
    parts = message.text.split()
    if len(parts) != 3:
        await message.reply("Используйте формат: /ban+[ID пользователя]+[количество дней]\nПример: /ban 123456789 5")
        return
    try:
        target_user = int(parts[1])
        days = int(parts[2])
        block_seconds = days * 24 * 3600 if days > 0 else 0
        banned_until = int(time.time()) + block_seconds if days > 0 else 0
        try:
            chat_member = await bot.get_chat_member(GROUP_ID, target_user)
            first_name = chat_member.user.first_name
        except Exception:
            first_name = "Неизвестно"
        await bot.restrict_chat_member(
            chat_id=GROUP_ID,
            user_id=target_user,
            permissions=types.ChatPermissions(can_send_messages=False),
            until_date=banned_until
        )
        user_link = f'<a href="tg://user?id={target_user}">{first_name}</a>'
        await message.reply(f"Пользователь {user_link} (ID: {target_user}) заблокирован {'навсегда' if days == 0 else f'на {days} дней'}.", parse_mode="HTML")
        add_ban(target_user, first_name, banned_until, "Ручная блокировка администратором")
        await notify_admins_about_ban(target_user, first_name, "Ручная блокировка администратором")
        # Отправляем уведомление в General о блокировке
        await notify_general(f"Пользователь {user_link} (ID: {target_user}) был заблокирован администратором.")
    except Exception as e:
        await message.reply(f"Ошибка: {e}")

@router.message(Command("unban"))
async def admin_unban(message: types.Message):
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in ADMIN_IDS:
        return
    parts = message.text.split()
    if len(parts) != 2:
        await message.reply("Используйте формат: /unban+[ID пользователя]\nПример: /unban 123456789")
        return
    try:
        target_user = int(parts[1])
        try:
            chat_member = await bot.get_chat_member(GROUP_ID, target_user)
            first_name = chat_member.user.first_name
        except Exception:
            first_name = "Неизвестно"
        await bot.restrict_chat_member(
            chat_id=GROUP_ID,
            user_id=target_user,
            permissions=types.ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
                can_send_polls=True,
                can_change_info=True,
                can_invite_users=True,
                can_pin_messages=True
            ),
            until_date=0
        )
        remove_ban(target_user)
        user_link = f'<a href="tg://user?id={target_user}">{first_name}</a>'
        await message.reply(f"Пользователь {user_link} (ID: {target_user}) успешно разблокирован.", parse_mode="HTML")
        # Отправляем уведомление в General о разблокировке
        await notify_general(f"Пользователь {user_link} (ID: {target_user}) был разблокирован администратором.")
    except Exception as e:
        await message.reply(f"Ошибка: {e}")

def add_ban(user_id: int, first_name: str, banned_until: int, reason: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO bans (user_id, first_name, banned_until, reason) VALUES (?, ?, ?, ?)",
        (user_id, first_name, banned_until, reason)
    )
    conn.commit()
    conn.close()

def remove_ban(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bans WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def get_banned_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = int(time.time())
    cursor.execute(
        "SELECT user_id, first_name, banned_until, reason FROM bans WHERE banned_until > ? OR banned_until = 0",
        (current_time,)
    )
    result = cursor.fetchall()
    conn.close()
    return result

@router.message(Command("admin"))
async def admin_panel(message: types.Message):
    if str(message.from_user.id) not in ADMIN_IDS:
        await message.reply("У вас нет доступа к этой команде.")
        return

    banned_users = get_banned_users()
    admin_text = (
        "Добро пожаловать в панель администратора!\n\n"
        "<b>Доступные команды:</b>\n"
        "<b>Команда:</b> <code>/ban+[ID пользователя]+[количество дней]</code>\n"
        "— Заблокировать пользователя на указанное количество дней. Если указать 0 дней, блокировка будет на неопределенное время (до момента разблокировки).\n"
        "<b>Пример:</b> <code>/ban 123456789 5</code>\n\n"
        "<b>Команда:</b> <code>/unban+[ID пользователя]</code>\n"
        "— Разблокировать пользователя.\n"
        "<b>Пример:</b> <code>/unban 123456789</code>\n\n"
        "При блокировке пользователя ботом (автоматически или вручную) пользователь продолжает оставаться в группе и имеет возможность просматривать сообщения, но лишается права их отправлять.\n\n"
        "<b>Команда:</b> <code>/topics</code>\n"
        "— Открыть настройки тем.\n"
    )

    if banned_users:
        admin_text += "\n<b>Заблокированные пользователи:</b>\n"
        for user in banned_users:
            user_id, first_name, banned_until, reason = user
            user_link = f'<a href="tg://user?id={user_id}">{first_name}</a>'
            days_left = "навсегда" if banned_until == 0 else round((banned_until - int(time.time())) / (24 * 3600))
            admin_text += f"- {user_link} (ID: {user_id}) — {days_left} дн., причина: {reason}\n"
    else:
        admin_text += "\n<b>Заблокированных пользователей нет.</b>\n"
    await message.reply(admin_text, parse_mode="HTML")

@router.message(Command("topics"))
async def list_topics(message: types.Message):
    # Команда должна вызываться в личном чате
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in [str(admin) for admin in ADMIN_IDS]:
        return

    # Получаем список тем из базы данных
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, enabled, block_days, warnings_limit, ad_frequency_days FROM topics")
    topics = cursor.fetchall()
    conn.close()

    if not topics:
        await message.reply("Темы не найдены.")
        return

    # Формируем описание доступных команд
    commands_info = (
        "📋 <b>Панель управления темами</b>\n\n"
        "<b>Доступные команды:</b>\n"
        "• <code>/switch+[ID темы]</code> — переключает состояние темы (включено/выключено).\n"
        "Пример: <code>/switch 5</code>\n"
        "• <code>/message+[ID темы]</code> — отправляет тестовое сообщение в указанную тему.\n"
        "Пример: <code>/message 5</code>\n"
        "• <code>/btime+[ID темы]+[кол-во дней от 0 до 365]</code> — устанавливает время блокировки для темы.\n"
        "0 дней означает блокировку <i>навсегда</i>.\n"
        "Пример: <code>/btime 5 10</code>\n"
        "• <code>/cwarn+[ID темы]+[кол-во предупреждений от 1 до 10]</code> — устанавливает число предупреждений до блокировки.\n"
        "1 предупреждение — блокировка происходит немедленно (при первом предупреждении).\n"
        "Пример: <code>/cwarn 5 3</code>\n"
        "• <code>/sdays+[ID темы]+[кол-во дней от 1 до 10]</code> — указывает минимальный допустимый интервал между рекламными объявлениями одного типа от одного пользователя.\n"
        "Пример: <code>/sdays 0 5</code>\n\n"
        "<b>Сводка по темам:</b>\n\n"
    )

    # Формируем сводку по каждой теме
    topics_summary = ""
    for t in topics:
        topic_id, enabled, block_days, warnings_limit, ad_frequency_days = t
        state = "🟢 Включена" if enabled else "🔴 Выключена"
        ad_interval = f"{ad_frequency_days} дней"
        block_time = "навсегда" if block_days == 0 else f"{block_days} дней"
        warn_text = "1 (немедленно)" if warnings_limit == 1 else f"{warnings_limit}"
        topics_summary += (
            f"<b>Тема (ID):</b> {topic_id}\n"
            f"<b>Состояние:</b> {state}\n"
            f"<b>Интервал между рекламными объявлениями:</b> {ad_interval}\n"
            f"<b>Время блокировки:</b> {block_time}\n"
            f"<b>Предупреждений до блокировки:</b> {warn_text}\n\n"
        )

    full_text = commands_info + topics_summary
    await message.reply(full_text, parse_mode="HTML")

def get_topic_status(topic_id: int):
    """Возвращает статус темы: 1 — включена, 0 — выключена, None — тема не найдена."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT enabled FROM topics WHERE thread_id=?", (topic_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def toggle_topic_status(topic_id: int):
    """Переключает статус темы: если включена – выключает, если выключена – включает.
       Возвращает новый статус или None, если тема не найдена."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT enabled FROM topics WHERE thread_id=?", (topic_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return None
    current_status = result[0]
    new_status = 0 if current_status else 1
    cursor.execute("UPDATE topics SET enabled=? WHERE thread_id=?", (new_status, topic_id))
    conn.commit()
    conn.close()
    return new_status

@router.message(Command("switch"))
async def switch_topic_handler(message: types.Message):
    # Если команда вызвана не в личном чате, игнорируем её.
    if message.chat.type != "private":
        return

    # Проверка: только админы могут использовать эту команду.
    if str(message.from_user.id) not in ADMIN_IDS:
        return

    # Извлекаем аргументы из сообщения (ожидается формат: /switch [номер темы])
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip().isdigit():
        await message.reply("Использование: /switch+[номер темы]. Пример: /switch 5")
        return

    topic_id = int(parts[1].strip())
    current_status = get_topic_status(topic_id)
    if current_status is None:
        await message.reply("Тема не найдена.")
        return

    new_status = toggle_topic_status(topic_id)
    status_text = "включена 🟢" if new_status == 1 else "выключена 🔴"
    await message.reply(f"Тема {topic_id} теперь {status_text}.")
    await list_topics(message)

@router.message(Command("message"))
async def send_test_message_handler(message: types.Message):
    # Команда должна вызываться в личном чате
    if message.chat.type != "private":
        return

    # Проверка на права администратора
    if str(message.from_user.id) not in ADMIN_IDS:
        return

    # Извлекаем аргументы (ожидается формат: /message [номер темы])
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip().isdigit():
        await message.reply("Использование: /message+[номер темы]. Пример: /message 5")
        return
    topic_id = int(parts[1].strip())
    current_status = get_topic_status(topic_id)
    if current_status is None:
        await message.reply("Тема не найдена.")
        return

    thread_id = int(parts[1].strip())
    # Формируем тестовое сообщение, которое не выглядит тупо, но информирует об идентификаторе темы.
    test_text = f"Проверка корректности работы темы. Идентификатор (ID) этой темы: {thread_id}."

    try:
        # Отправляем тестовое сообщение в указанную тему группы
        await bot.send_message(chat_id=GROUP_ID, message_thread_id=thread_id, text=test_text)
        await message.reply(f"Тестовое сообщение успешно отправлено в тему с идентификатором (ID) {thread_id}.")
        await list_topics(message)
    except Exception as e:
        await message.reply(f"Ошибка при отправке тестового сообщения: {e}")

def update_topic_block_days(topic_id: int, days: int) -> bool:
    """Обновляет время блокировки (block_days) в таблице topics для заданной темы.
       Возвращает True при успехе, False если тема не найдена."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT block_days FROM topics WHERE thread_id=?", (topic_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False
    cursor.execute("UPDATE topics SET block_days=? WHERE thread_id=?", (days, topic_id))
    conn.commit()
    conn.close()
    return True

def update_topic_warnings_limit(topic_id: int, warnings_limit: int) -> bool:
    """Обновляет количество предупреждений до блокировки (warnings_limit) в таблице topics для заданной темы.
       Возвращает True при успехе, False если тема не найдена."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT warnings_limit FROM topics WHERE thread_id=?", (topic_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False
    cursor.execute("UPDATE topics SET warnings_limit=? WHERE thread_id=?", (warnings_limit, topic_id))
    conn.commit()
    conn.close()
    return True


def create_summary_text(topic_id: int) -> str:
    settings = get_topic_settings(topic_id)
    if not settings:
        return "Не удалось получить настройки темы."
    block_days = settings["block_days"]
    warnings_limit = settings["warnings_limit"]
    ad_frequency = settings["ad_frequency_days"]
    time_text = "навсегда" if block_days == 0 else f"{block_days} дней"

    if warnings_limit == 1:
        warn_text = "немедленно (при первом предупреждении)"
    else:
        warn_text = f"после получения {warnings_limit} предупреждений"

    summary = (f"Сводка настроек темы {topic_id}:\n"
               f"• Пользователи могут размещать один тип рекламы не чаще, чем раз в {ad_frequency} дней.\n"
               f"• Время блокировки: {time_text}\n"
               f"• Блокировка происходит {warn_text}.\n\n"
               f"Это означает, что при накоплении указанного количества предупреждений пользователь будет заблокирован, "
               f"то есть не сможет отправлять сообщения, а только читать их в группе в течение указанного срока блокировки или навсегда.")
    return summary


@router.message(Command("btime"))
async def set_block_time_handler(message: types.Message):
    # Команда должна вызываться в личном чате.
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in ADMIN_IDS:
        return

    parts = message.text.split()
    if len(parts) != 3:
        await message.reply("Использование: /btime+[ID темы]+[количество дней от 0 до 365]. Пример: /btime 5 10")
        return

    topic_id_str, days_str = parts[1], parts[2]
    if not topic_id_str.isdigit() or not days_str.isdigit():
        await message.reply("ID темы и количество дней должны быть числовыми значениями.")
        return

    topic_id = int(topic_id_str)
    days = int(days_str)
    if days < 0 or days > 365:
        await message.reply("Количество дней должно быть в диапазоне от 0 до 365.")
        return

    success = update_topic_block_days(topic_id, days)
    if not success:
        await message.reply("Тема не найдена.")
        return

    time_text = "навсегда" if days == 0 else f"{days} дней"
    await message.reply(f"В теме {topic_id} время блокировки установлено {time_text}.")
    await list_topics(message)

    summary = create_summary_text(topic_id)
    try:
        await bot.send_message(
            chat_id=GROUP_ID,
            message_thread_id=topic_id,
            text=f"Настройки темы изменены.\n{summary}"
        )
    except Exception as e:
        print(f"Ошибка отправки уведомления в тему {topic_id}: {e}")


@router.message(Command("cwarn"))
async def set_warnings_limit_handler(message: types.Message):
    # Команда должна вызываться в личном чате.
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in ADMIN_IDS:
        return

    parts = message.text.split()
    if len(parts) != 3:
        await message.reply("Использование: /cwarn+[ID темы]+[количество предупреждений до блокировки (от 1 до 10)]. Пример: /cwarn 5 3")
        return

    topic_id_str, warn_str = parts[1], parts[2]
    if not topic_id_str.isdigit() or not warn_str.isdigit():
        await message.reply("ID темы и количество предупреждений должны быть числовыми значениями.")
        return

    topic_id = int(topic_id_str)
    warnings_limit = int(warn_str)
    if warnings_limit < 1 or warnings_limit > 10:
        await message.reply("Количество предупреждений должно быть от 1 до 10.")
        return

    success = update_topic_warnings_limit(topic_id, warnings_limit)
    if not success:
        await message.reply("Тема не найдена.")
        return

    await message.reply(f"В теме {topic_id} количество предупреждений до блокировки установлено на {warnings_limit}.")
    await list_topics(message)

    summary = create_summary_text(topic_id)
    try:
        await bot.send_message(
            chat_id=GROUP_ID,
            message_thread_id=topic_id,
            text=f"Настройки темы изменены.\n{summary}"
        )
    except Exception as e:
        print(f"Ошибка отправки уведомления в тему {topic_id}: {e}")

@router.message(Command("sdays"))
async def set_ad_frequency(message: types.Message):
    # Команда должна вызываться в личном чате.
    if message.chat.type != "private":
        return
    if str(message.from_user.id) not in ADMIN_IDS:
        return

    # Разбираем аргументы команды
    parts = message.text.split()
    if len(parts) !=3:
        await message.reply("Используйте: /sdays+[ID темы]+[количество дней от 1 до 10]. Пример: /sdays 0 5")
        return

    thread_id = parts[1]
    if not thread_id.isdigit():
        await message.reply("ID темы должно быть числом.")
        return

    # Устанавливаем значение по умолчанию (5 дней)
    days = 5
    if len(parts) == 3:
        if not parts[2].isdigit() or not (1 <= int(parts[2]) <= 10):
            await message.reply("Количество дней должно быть от 1 до 10.")
            return
        days = int(parts[2])

    # Обновляем базу данных
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE topics SET ad_frequency_days=? WHERE thread_id=?",
        (days, thread_id)
    )
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        await message.reply(f"Тема с ID {thread_id} не найдена.")
    else:
        await message.reply(f"Периодичность рекламы для темы {thread_id} установлена на {days} дней.")
        await list_topics(message)

        summary = create_summary_text(int(thread_id))
        try:
            await bot.send_message(
                chat_id=GROUP_ID,
                message_thread_id=thread_id,
                text=f"Настройки темы изменены.\n{summary}"
            )
        except Exception as e:
            print(f"Ошибка отправки уведомления в тему {thread_id}: {e}")
