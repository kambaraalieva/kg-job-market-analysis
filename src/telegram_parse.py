"""Free-text parser for kgworkzone-style vacancy posts. Used by notebook 07."""
import re
import numpy as np
import pandas as pd

EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍•·▪️◾️🔹🔸]+")
HEADER = re.compile(r"^(?:вакансии?|ваканс\w*|требуетс[яь]|требуются|срочно|внимание|ищем|набор|жумуш\w*)\W*$", re.I)
SAL_LINE = re.compile(r"(зарплат\w*|заработн\w* плат\w*|оплат\w*|з/п|\bзп\b|айлык\w*|доход\w*|оклад\w*|💰)", re.I)
NUM = re.compile(r"\d[\d\s .,]*\d|\d")


def clean_line(s):
    return EMOJI.sub(" ", s).replace("*", " ").strip(" -–—:|\t")


def split_blocks(text):
    """Posts with several vacancies ('🔹 Охранник ... 💰 ...') are split into blocks."""
    parts = re.split(r"\n\s*(?=🔹|🔸|▪️|◾)", text)
    blocks = [p for p in parts if SAL_LINE.search(p)]
    if len(blocks) >= 2:
        return blocks
    return [text]


PREFIX = re.compile(r"^(?:актуальн\w*\s+)?(?:ваканси[яи]\s*[:\-–]?\s*)|(?:^.*?\bтребу(?:ется|ются|етьс[яь])\b\s*[:\-–]?\s*)", re.I)


def title_of(block):
    lines = block.split("\n")
    for line in lines:
        c = clean_line(line)
        if len(c) < 3 or HEADER.match(c) or SAL_LINE.search(line) or c.lower().startswith(("наш канал", "размещ")):
            continue
        rest = PREFIX.sub("", c).strip(" :–-")
        if len(rest) < 3:          # «В ресторан требуются» — сама должность на следующей строке
            continue
        return rest[:1].upper() + rest[1:120]
    return ""


VACANCY = re.compile(r"требу|ваканси|ищем|приглаша\w* (?:на работу|в команду|кассир|продав|офици|повар|сотрудник)|"
                     r"нужен|нужна\b|нужны\b|керек|набор|обязанност|требовани|💰", re.I)
NOT_VACANCY = re.compile(r"ищу работу|резюме\s*:|опрос|реклам\w* казино|как вы думаете|наш канал создан|мы создали|"
                         r"напоминание|в закрепленн|выходные\s*[—-]|фильм\s*:|если вам не хватает", re.I)


def is_vacancy(text):
    return bool(VACANCY.search(text)) and not NOT_VACANCY.search(text[:300])


def parse_salary(block):
    """Returns (from, to, period, currency, line). period: month/day/week/hour/None."""
    for line in block.split("\n"):
        if not SAL_LINE.search(line):
            continue
        low = line.lower()
        nums = [float(re.sub(r"[\s ]", "", n).replace(",", ".")) for n in NUM.findall(low)
                if re.sub(r"[\s .,]", "", n).isdigit()]
        nums = [n for n in nums if n >= 100]                     # отбрасывает "2 раза", "6/1", часы
        if "тыс" in low or re.search(r"\d\s*к\b", low):
            nums = [n * 1000 if n < 1000 else n for n in nums]
        if not nums:
            if re.search(r"договор|по результат|жараша|зависит|обсужд", low):
                return np.nan, np.nan, None, None, line
            continue
        cur = ("USD" if re.search(r"\$|usd|долл", low) else "EUR" if re.search(r"€|eur|евро", low)
               else "RUB" if re.search(r"₽|руб", low) else "KZT" if re.search(r"₸|тенге", low) else "KGS")
        if re.search(r"за смену|в смену|за выход|в день|/день|/смен|за день|күнүнө", low):
            period = "day"
        elif re.search(r"в час|/час|за час", low):
            period = "hour"
        elif re.search(r"в месяц|/\s*мес|за месяц|айына|в мес\b", low):
            period = "month"              # «в месяц» важнее «выплаты раз в неделю»
        elif re.search(r"за неделю|/недел", low):
            period = "week"
        else:
            period = None
        has_from = re.search(r"\bот\b", low) is not None
        has_to = re.search(r"\bдо\b", low) is not None
        if len(nums) >= 2 and (re.search(r"\d\s*[-–—]\s*\d", low) or (has_from and has_to)):
            return nums[0], nums[1], period, cur, line
        if has_to and not has_from:
            return np.nan, nums[0], period, cur, line
        return nums[0], np.nan, period, cur, line
    return np.nan, np.nan, None, None, ""


def experience_of(text):
    t = text.lower()
    if re.search(r"без опыта|опыт\w*[^\n.]{0,25}не (?:требуется|обязател|нужен|важен)|обучим|всему обучат|тажрыйба\w* талап кылынбайт", t):
        return 0
    m = re.search(r"опыт\w*[^\n]{0,40}?(?:от|не менее|более|минимум)\s*(\d+)\s*(год|лет|мес)", t)
    if m:
        n = int(m.group(1))
        if m.group(2).startswith("мес"):
            return 0 if n < 12 else 1
        return 0 if n < 1 else 1 if n < 3 else 3 if n < 6 else 6
    if re.search(r"опыт\w*[^\n]{0,30}(?:обязател|приветств|желател|работы)|с опытом", t):
        return 1
    return None


def city_of(text):
    t = text.lower()
    if re.search(r"\bош\b|г\.\s*ош|ошск\w* област", t) and "ошский рынок" not in t and "ошского рынка" not in t:
        return "Ош"
    if re.search(r"район\b|\bс\.\s*[а-я]|сокулук|кант\b|токмок|кара-балта|чолпон|каракол|джалал|нарын|талас|баткен", t):
        return "Регионы"
    return "Бишкек"        # канал «Работа Бишкек»: по умолчанию Бишкек


def parse_free(posts):
    rows = []
    posts = posts[posts["text"].map(is_vacancy)]
    for r in posts.itertuples():
        for k, block in enumerate(split_blocks(r.text)):
            f, t, period, cur, line = parse_salary(block)
            rows.append({
                "msg_id": r.msg_id, "block": k, "channel": r.channel, "link": r.link, "published": r.published,
                "name": title_of(block), "zp": line.strip(), "compensation_from": f, "compensation_to": t, "compensation_currency": cur,
                "period_word": period, "exp": experience_of(r.text if k == 0 else block + "\n" + r.text),
                "area": city_of(r.text), "text": r.text if len(split_blocks(r.text)) == 1 else block,
            })
    return pd.DataFrame(rows)
