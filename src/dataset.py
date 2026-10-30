"""
Master dataset of vacancies 2024-2026: one table, one set of rules for all sources.

Sources
    data/raw/vacancy.xlsx            hh.kg + lalafo.kg (School of Data Kyrgyzstan), snapshot Feb-Mar 2024
    data/raw/telegram_posts.jsonl    Telegram @codifynews (Jan 2024 - Oct 2026), @kgworkzone (Nov 2025 - Oct 2026)

Principles
    1. Nothing is invented. A value not stated in the ad stays empty (experience, city, education ...).
       Where a value is derived by a rule, a *_basis column says how.
    2. Dates are kept exactly, in Bishkek time (UTC+6).
    3. Every word of the ad is kept: full text in `ad_text`, everything not turned into a column in `additional_info`.
    4. Rows are removed only for a logged reason (not a vacancy, repost, outside 2024-2026).

Notebook 09_master_dataset runs `build()` and documents every rule with counts and examples.
"""
import json
import re

import numpy as np
import pandas as pd

import telegram_parse as tp

TZ = "Asia/Bishkek"
PERIOD_START = pd.Timestamp("2024-01-01")
REPOST_WINDOW_DAYS = 30
SALARY_MIN, SALARY_MAX = 10_000, 600_000

# NBKR rates, som per unit. RUB and KZT are not converted (see currency rule).
FX = {2024: ({"KGS": 1.0, "USD": 89.4870, "EUR": 97.7824}, "2024-03-14"),
      2025: ({"KGS": 1.0, "USD": 87.45, "EUR": 103.2347}, "2025-07-02"),
      2026: ({"KGS": 1.0, "USD": 87.45, "EUR": 99.6843}, "2026-06-30")}

SOURCE_NAME = {"headhunter": "hh.kg", "lalafo": "lalafo.kg",
               "codifynews": "Telegram @codifynews", "kgworkzone": "Telegram @kgworkzone"}
EXPERIENCE_BUCKET = {0: "без опыта", 1: "1–3 года", 3: "3–6 лет", 6: "6+ лет"}

LOG = []


def note(step, reason, rows):
    LOG.append({"шаг": step, "почему": reason, "строк": int(rows)})


# ======================================================================= dates
def to_bishkek(ts):
    """UTC timestamps (unix seconds or ISO strings) -> naive local time in Bishkek."""
    if np.issubdtype(np.asarray(ts).dtype, np.number):
        t = pd.to_datetime(ts, unit="s", utc=True)
    else:
        t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(TZ).dt.tz_localize(None)


# ======================================================================= text rules (all sources)
def norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^а-яёa-z0-9]+", " ", regex=True).str.strip()


def education_rule(t):
    return np.select([t.str.contains(r"высше\w* образован|высшее\b|бакалавр|магистр|жогорку билим"),
                      t.str.contains(r"средне\w*[- ]?специальн|среднее профессиональн|колледж")],
                     ["высшее", "среднее специальное"], default=None)


def gender_rule(t):
    female = t.str.contains(r"девушк|женщин|аял киши|кыздар|\bкыз\b")
    male = t.str.contains(r"\bпарн|мужчин|эркек|жигит")
    return np.select([female & ~male, male & ~female, female & male], ["женщины", "мужчины", "женщины или мужчины"],
                     default=None)


def schedule_rule(t):
    return t.str.extract(r"\b([1-7]/[1-7])\b")[0]


def commission_rule(t):
    return t.str.contains(r"процент|% от|бонус")


LEVELS = [("lead", r"\b(lead|тимлид|team ?lead|ведущ\w*|head|руководител\w*|principal|architect|архитектор)\b"),
          ("senior", r"\b(senior|сеньор|синьор|старш\w*|sr)\b"),
          ("middle", r"\b(middle|мидл|mid)\b"),
          ("junior", r"\b(junior|джун\w*|младш\w*|jr|стаж[её]р\w*|intern|trainee|ассистент)\b")]


def level_in_title(title):
    t = str(title).lower()
    for name, pattern in LEVELS:
        if re.search(pattern, t):
            return name
    return None


def work_format_rule(type_text, text):
    """codify: the «Тип» field; other sources: words in the ad."""
    t = (str(type_text) if isinstance(type_text, str) else "").lower()
    if t:
        if re.search(r"стаж|intern", t): return "стажировка"
        if "project" in t or "проект" in t: return "проектная работа"
        if re.search(r"частичн|неполный|part", t): return "частичная занятость"
        if re.search(r"remote|удал", t): return "удалённо или гибрид"
        if re.search(r"office|полная|полный|full", t): return "офис, полный день"
    x = str(text).lower()
    if re.search(r"удал[её]нн|remote|онлайн работ", x): return "удалённо или гибрид"
    return None


# ======================================================================= experience
def experience_from_text(text):
    """Explicit statements only. Returns (bucket, min_years) or (None, None)."""
    t = str(text).lower()
    if re.search(r"без опыта|опыт\w*[^\n.]{0,25}не (?:требуется|обязател|нужен|важен)|обучим|всему обучат|"
                 r"тажрыйба\w* талап кылынбайт", t):
        return "без опыта", 0
    m = re.search(r"опыт\w*[^\n]{0,40}?(?:от|не менее|более|минимум)\s*(\d+)\s*(год|лет|мес)", t)
    if m:
        n = int(m.group(1))
        years = n / 12 if m.group(2).startswith("мес") else n
        k = 0 if years < 1 else 1 if years < 3 else 3 if years < 6 else 6
        return EXPERIENCE_BUCKET[k], k
    if re.search(r"опыт\w*[^\n]{0,30}(?:обязател|приветств|желател)|с опытом|опыт работы", t):
        return "требуется, срок не указан", None
    return None, None


# ======================================================================= salary
PERIOD_WORDS = {"hour": r"в час\b|за час|/\s*час|почасов",
                "day": r"за смену|в смену|за выход|в день|/\s*день|/\s*смен|за день|күнүнө",
                "week": r"в неделю|за неделю|/\s*недел|еженедельн|жумасына",
                "month": r"в месяц|/\s*мес|за месяц|айына|в мес\b"}
PERIOD_FACTOR = {"hour": 8 * 26, "day": 26, "week": 4.33, "month": 1}
PERIOD_RU = {"hour": "час", "day": "день", "week": "неделя", "month": "месяц"}


def pay_period_rule(amount, text):
    """Returns (period, basis). `text`: the salary line (Telegram) or the whole ad (hh, lalafo)."""
    t = str(text).lower()
    # «выплаты 2 раза в месяц», «выдаётся раз в неделю»: это частота выплат, а не период ставки
    t = re.sub(r"(?:выда\w*|выплат\w*|выплачива\w*|зп)?[^.,;()]{0,12}(?:\d|один|два|две|три|1|2)?\s*раз\w*\s+в\s+(?:месяц|неделю)", " ", t)
    said = {p for p, rx in PERIOD_WORDS.items() if re.search(rx, t)}
    if "hour" in said and amount < 1000:
        return "hour", "указано в тексте"
    if "day" in said and amount < 10_000:
        return "day", "указано в тексте"
    if "month" in said:
        return "month", "указано в тексте"
    if "week" in said and 5_000 <= amount < 30_000:
        return "week", "указано в тексте"
    if 500 <= amount < 5_000:
        return "day", "по сумме: 500–5 000 сом в месяц неправдоподобно"
    return "month", "не указано, считается месячной"


def salary_columns(df, period_text):
    """Adds salary_basis, pay_period, salary_month_kgs, salary_status, fx columns. Uses salary_from/to/currency."""
    f, t, cur = df["salary_from"], df["salary_to"], df["currency"]
    has = f.notna() | t.notna()
    df["salary_basis"] = np.select([f.notna() & t.notna(), f.notna(), t.notna()],
                                   ["середина диапазона", "нижняя граница («от»)", "верхняя граница («до»)"], default=None)
    amount = pd.Series(np.where(f.notna() & t.notna(), (f + t) / 2, f.fillna(t)), index=df.index)
    year = df["reference_date"].dt.year.clip(2024, 2026)
    rate = [FX[y][0].get(c, np.nan) if isinstance(c, str) else np.nan for y, c in zip(year, cur)]
    df["fx_rate"] = np.where(cur.isin(["USD", "EUR"]), rate, np.nan)
    df["fx_rate_date"] = np.where(cur.isin(["USD", "EUR"]), [FX[y][1] for y in year], None)
    kgs = amount * np.array(rate, dtype=float)

    periods = [pay_period_rule(a, x) if pd.notna(a) else (None, None) for a, x in zip(kgs, period_text)]
    df["pay_period"] = [PERIOD_RU.get(p) for p, _ in periods]
    df["pay_period_basis"] = [b for _, b in periods]
    month = kgs * np.array([PERIOD_FACTOR.get(p, np.nan) if p else np.nan for p, _ in periods], dtype=float)

    foreign = has & cur.isin(["RUB", "KZT"])
    implausible = has & ~foreign & pd.Series(month, index=df.index).notna() & ((month < SALARY_MIN) | (month > SALARY_MAX))
    unpaid = df.get("unpaid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["salary_status"] = np.select(
        [unpaid, ~has, foreign, implausible],
        ["неоплачиваемая", "скрыта", "в рублях или тенге, не пересчитана", "неправдоподобная (меньше 10 тыс. или больше 600 тыс.)"],
        default="указана")
    df["salary_month_kgs"] = np.where(df["salary_status"] == "указана", np.round(month, -1), np.nan)
    return df


def codify_salary(zp):
    """«ЗП: От 30000 до 70000 KGS в месяц» -> (from, to, currency)."""
    z = str(zp) if isinstance(zp, str) else ""
    m = re.search(r"(KGS|USD|EUR|RUB|RUR|KZT|₽|\$|€|₸|сом)", z, re.I)
    cur = {"$": "USD", "€": "EUR", "₽": "RUB", "RUR": "RUB", "₸": "KZT", "СОМ": "KGS"}.get(m.group(1).upper(), m.group(1).upper()) if m else None
    nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"(?<=\d)[\s ](?=\d{3})", "", z))]
    if not nums:
        return np.nan, np.nan, None
    lo = re.match(r"^\s*от", z, re.I) and not re.search(r"\bдо\b", z, re.I)
    hi = re.match(r"^\s*до", z, re.I)
    if hi:
        return np.nan, nums[0], cur or "KGS"
    if lo or len(nums) == 1:
        return nums[0], np.nan, cur or "KGS"
    return nums[0], nums[1], cur or "KGS"


CITY_GROUP = lambda a: (None if a is None or (isinstance(a, float) and np.isnan(a))
                        else "Бишкек" if a == "Бишкек" else "Ош" if a == "Ош" else "Регионы")


def city_from_text(text):
    """Only explicit mentions; nothing is assumed from the channel name."""
    t = str(text).lower()
    if re.search(r"\bош\b|г\.\s*ош|ошск\w* област", t) and "ошский рынок" not in t and "ошского рынка" not in t:
        return "Ош"
    m = re.search(r"\b(сокулук|кант|токмок|кара-балта|чолпон-ата|каракол|джалал-абад|нарын|талас|баткен|лебединовка|"
                  r"беловодское|кемин|балыкчы|узген|кара-суу)\b", t)
    if m:
        return m.group(1).title()
    if re.search(r"бишкек|г\.\s*бишкек", t):
        return "Бишкек"
    return None


NOT_VACANCY_EXTRA = re.compile(r"наш канал созда|работодатель среди подписчиков|канал создаётся|поддержка реально помогает|"
                               r"делиться каналом", re.I)
CHANNEL_TITLE = re.compile(r"канал", re.I)     # заголовок про сам канал, а не про вакансию


CUT = re.compile(r"📍|адрес|\bул\.|\bтел\b|телефон|ватсап|whatsapp|\[телефон\]|\[контакт\]", re.I)
MONEY = re.compile(r"\d[\d\s]*\s*(?:сом|с\b|kgs|\$|usd|долл|тыс|к\b|руб|тенге)|\d[\d\s]{2,}\s*[-–—]\s*\d", re.I)


def _clean_for_salary(block):
    """Phones are hidden and the part of a line after «📍 / Адрес / ул. / телефон» is cut,
    so house and phone numbers are not read as salaries."""
    out = []
    for ln in mask_phones(block).split("\n"):
        m = CUT.search(ln)
        out.append(ln[:m.start()] if m else ln)
    return "\n".join(out)


def salary_of_block(block):
    """tp.parse_salary on the cleaned block, plus: if the salary line («💰 Оплата:») has no numbers,
    the next lines that look like money are read (preferring one that says «в месяц»)."""
    block = _clean_for_salary(block)
    f, t, period, cur, line = tp.parse_salary(block)
    if not (np.isnan(f) and np.isnan(t)):
        return f, t, period, cur, line
    lines = block.split("\n")
    i = next((j for j, ln in enumerate(lines) if tp.SAL_LINE.search(ln) and not re.search(r"\d", ln)), None)
    if i is None or re.search(r"договор|обсужд|по результат|жараша|зависит", lines[i], re.I):
        return f, t, period, cur, line
    nxt = [ln for ln in lines[i + 1:i + 5] if MONEY.search(ln)]
    nxt = sorted(nxt, key=lambda ln: 0 if re.search(PERIOD_WORDS["month"], ln.lower()) else 1)
    for ln in nxt:
        f2, t2, p2, c2, _ = tp.parse_salary("Зарплата: " + ln)
        if not (np.isnan(f2) and np.isnan(t2)):
            return f2, t2, p2, c2, lines[i].strip() + " " + ln.strip()
    return f, t, period, cur, line


PHONE = [re.compile(p) for p in [r"wa\.me/\d+", r"\+\d{1,3}[\s\-()]*\d[\d\s\-()]{7,14}\d", r"(?<!\d)\d{10,12}(?!\d)",r"\+?996[\s\-()]*\d{3}[\s\-()]*\d{2,3}[\s\-]*\d{2,3}[\s\-]*\d{0,3}",
                                  r"(?<!\d)0\d{3}[\s\-]?\d{2,3}[\s\-]?\d{2,3}[\s\-]?\d{0,3}(?!\d)",
                                  r"(?<!\d)[2579]\d{8}(?!\d)"]]


def mask_phones(s):
    if not isinstance(s, str):
        return s
    for p in PHONE:
        s = p.sub("[телефон]", s)
    return s


# ======================================================================= sources
def load_hh_lalafo(path):
    df = pd.read_excel(path)
    n0 = len(df)
    df["text_lc"] = (df["name"].fillna("") + " " + df["requirements"].fillna("") + " " + df["responsibilities"].fillna("")).str.lower()
    seeker = df["text_lc"].str.contains(r"\bищу работу|\bищу подработку|иш издейм|иш издеп жатам")
    service = df["text_lc"].str.contains(r"возьму в аренду|\bсдаю\b|\bсдается\b|\bпродаю\b|\bпрода[её]тся\b|отремонтируем")
    note("hh/lalafo: удалены резюме и объявления об услугах", "не вакансии", (seeker | service).sum())
    df = df[~(seeker | service)].copy()

    out = pd.DataFrame(index=df.index)
    out["vacancy_id"] = df["source"].map({"headhunter": "hh_", "lalafo": "lalafo_"}) + df["id"].astype(str)
    out["source_key"] = df["source"]
    out["url"] = df["link"]
    out["published_at"] = to_bishkek(df["publication_time"])
    out["last_updated_at"] = to_bishkek(df["last_change_time"])
    # hh: date of publication. lalafo: ads are re-raised; the last update shows when the ad was live
    out["reference_date"] = out["last_updated_at"].where(df["source"] == "lalafo", out["published_at"])
    out["date_basis"] = np.where(df["source"] == "lalafo", "последнее обновление объявления", "публикация")
    out["job_title"] = df["name"].str.strip()
    out["employer"] = df["company"]
    out["sector"] = df["category"]
    out["profession"] = df["sub_category"]
    out["profession_basis"] = "указана источником"
    out["experience_required"] = df["experience_start"].map(EXPERIENCE_BUCKET)
    out["experience_min_years"] = df["experience_start"]
    out["experience_basis"] = "поле сайта"
    out["level_in_post"] = None
    out["work_format_raw"] = None
    out["locality"] = df["area"]
    out["address"] = df["address_display_name"]
    out["salary_text"] = None
    out["salary_from"] = df["compensation_from"]
    out["salary_to"] = df["compensation_to"]
    cur = df["compensation_currency"].replace({"RUR": "RUB"})
    no_cur = (df["compensation_from"].notna() | df["compensation_to"].notna()) & cur.isna()
    note("hh/lalafo: валюта не указана → KGS", "объявления lalafo из Кыргызстана", no_cur.sum())
    out["currency"] = cur.mask(no_cur, "KGS")
    out["skills"] = df["skills"]
    out["ad_text"] = (df["requirements"].fillna("") + "\n" + df["responsibilities"].fillna("")).str.strip()
    out["additional_info"] = out["ad_text"]
    out["period_text"] = df["text_lc"]
    out["unpaid"] = False
    print(f"hh/lalafo: {n0} строк в файле, {len(out)} после удаления не-вакансий")
    return out.reset_index(drop=True)


def load_telegram(path):
    raw = pd.read_json(path, lines=True, dtype={"msg_id": str, "channel": str})
    raw["published_at"] = to_bishkek(raw["date"])
    other = ~raw["channel"].isin(["codifynews", "kgworkzone"])
    note("Telegram: посты других каналов", "DevKG: 20 постов в другом формате", other.sum())
    raw = raw[~other]

    # ---------- codifynews: template posts
    cod = raw[raw["channel"] == "codifynews"]
    tpl = cod["text"].fillna("").str.contains("Компания:") & cod["text"].fillna("").str.contains("ЗП:")
    note("codifynews: посты не по шаблону", "реклама, анонсы, подкасты", (~tpl).sum())
    rows = []
    for r in cod[tpl].itertuples():
        lines = [x.strip() for x in r.text.split("\n")]
        field = lambda k: next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith(k + ":")), None)
        f, t, cur = codify_salary(field("ЗП"))
        extra = field("Дополнительно")
        type_ = field("Тип")
        exp_txt = re.match(r"^(Без опыта|От 1 до 3 лет|От 3 до 6 лет|Более 6 лет)", extra or "")
        exp_k = {"Без опыта": 0, "От 1 до 3 лет": 1, "От 3 до 6 лет": 3, "Более 6 лет": 6}.get(exp_txt.group(1)) if exp_txt else None
        area = re.search(r",\s*([^,]+)$", extra or "")
        lvl = re.search(r"(стажёр|младший|средний|старший|ведущий)\s*\((\w+)\)", (type_ or "").lower())
        info = [ln for ln in lines[1:] if ln and not ln.startswith(("Компания:", "ЗП:"))]
        rows.append({
            "vacancy_id": f"tg_codifynews_{r.msg_id}", "source_key": "codifynews", "url": r.link,
            "published_at": r.published_at, "job_title": lines[0], "employer": field("Компания"),
            "salary_text": field("ЗП"), "salary_from": f, "salary_to": t, "currency": cur,
            "unpaid": bool(re.search(r"неоплачиваем", field("ЗП") or "", re.I)),
            "experience_required": EXPERIENCE_BUCKET.get(exp_k), "experience_min_years": exp_k,
            "experience_basis": "поле «Дополнительно»" if exp_k is not None else None,
            "level_in_post": lvl.group(2) if lvl else None, "work_format_raw": type_,
            "locality": area.group(1).strip() if area else None, "address": None,
            "ad_text": r.text, "additional_info": "\n".join(info) or None, "period_text": field("ЗП") or "",
        })
    c = pd.DataFrame(rows)

    # ---------- kgworkzone: free text, possibly several vacancies per post
    kz_posts = raw[raw["channel"] == "kgworkzone"]
    vac = kz_posts["text"].fillna("").map(lambda x: tp.is_vacancy(x) and not NOT_VACANCY_EXTRA.search(x))
    note("kgworkzone: посты не про вакансии", "мемы, советы, анонсы канала", (~vac).sum())
    rows = []
    for r in kz_posts[vac].itertuples():
        blocks = tp.split_blocks(r.text)
        for k, block in enumerate(blocks):
            f, t, period, cur, sal_line = salary_of_block(block)
            title = tp.title_of(block)
            ctx = block if len(blocks) > 1 else r.text
            bucket, years = experience_from_text(ctx if len(blocks) == 1 else block + "\n" + r.text)
            title_line = next((ln for ln in block.split("\n") if title and title[:15].lower() in ln.lower()), None)
            used = set(sal_line.split("\n")) | {title_line}
            info = [ln.strip() for ln in ctx.split("\n") if ln.strip() and ln not in used and ln.strip() not in sal_line]
            locality = city_from_text(r.text)
            rows.append({
                "vacancy_id": f"tg_kgworkzone_{r.msg_id}_{k}", "source_key": "kgworkzone", "url": r.link,
                "published_at": r.published_at, "job_title": title or None, "employer": None,
                "salary_text": sal_line.strip() or None, "salary_from": f, "salary_to": t,
                "currency": {"RUB": "RUB", "KZT": "KZT"}.get(cur, cur), "unpaid": False,
                "experience_required": bucket, "experience_min_years": years,
                "experience_basis": "текст объявления" if bucket else None, "level_in_post": None,
                "work_format_raw": None, "locality": locality, "address": None,
                "ad_text": ctx, "additional_info": "\n".join(info) or None, "period_text": sal_line or "",
            })
    k = pd.DataFrame(rows)
    promo = k["job_title"].fillna("").str.contains(CHANNEL_TITLE) | k["job_title"].isna()
    note("kgworkzone: заметки о канале и посты без заголовка", "заголовок про канал («этот канал…») или не найден", promo.sum())
    k = k[~promo]
    note("kgworkzone: посты с несколькими вакансиями разделены", "одна строка на вакансию",
         (k["vacancy_id"].str.rsplit("_", n=1).str[0].duplicated(keep=False)).sum())

    tg = pd.concat([c, k], ignore_index=True)
    tg["last_updated_at"] = pd.NaT
    tg["reference_date"] = tg["published_at"]
    tg["date_basis"] = "публикация в канале"
    tg["skills"] = None
    tg["profession_basis"] = "классификатор по заголовку"
    print(f"Telegram: {len(raw)} постов → {len(c)} вакансий codifynews, {len(k)} вакансий kgworkzone")
    return tg


# ======================================================================= profession classifier
def classify_professions(tg, train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.svm import LinearSVC

    def clf():
        return make_pipeline(make_union(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)), LinearSVC(C=0.5))

    x = train["job_title"].fillna("").str.lower()
    grp = x.str.replace(r"[^а-яёa-z]+", " ", regex=True)
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(train, groups=grp))
    acc = {}
    for col in ["sector", "profession"]:
        acc[col] = float((clf().fit(x.iloc[a], train[col].iloc[a]).predict(x.iloc[b]) == train[col].iloc[b]).mean())
        tg[col] = clf().fit(x, train[col]).predict(tg["job_title"].fillna("").str.lower())
    return tg, acc


# ======================================================================= build
def build(xlsx="../data/raw/vacancy.xlsx", jsonl="../data/raw/telegram_posts.jsonl"):
    LOG.clear()
    hl = load_hh_lalafo(xlsx)
    SALES = "Менеджер по продажам, менеджер по работе с клиентами"
    wrong = (hl["profession"] == SALES) & (hl["sector"] != "Продажи, обслуживание клиентов")
    note("Профессия «Менеджер по продажам» перенесена в сферу «Продажи»", "ошибка справочника источника", wrong.sum())
    hl.loc[wrong, "sector"] = "Продажи, обслуживание клиентов"

    tg = load_telegram(jsonl)
    tg, acc = classify_professions(tg, hl)
    d = pd.concat([hl, tg], ignore_index=True)

    # ---------- period
    old = d["reference_date"] < PERIOD_START
    note("Вне периода 2024–2026", "lalafo: объявление не обновлялось с 2023 года", old.sum())
    d = d[~old].copy()

    # ---------- reposts: same source + title + employer + salary text/amount + start of text within 30 days
    d["_key"] = (d["source_key"] + "|" + norm(d["job_title"]) + "|" + norm(d["employer"]) + "|"
                 + norm(d["salary_text"].fillna(d["salary_from"].astype(str) + "-" + d["salary_to"].astype(str)))
                 + "|" + norm(d["ad_text"]).str.slice(0, 120).where(d["source_key"] != "codifynews", ""))
    d = d.sort_values("reference_date").reset_index(drop=True)
    first_of_run, run_id, last_seen = {}, [], {}
    for i, (key, when) in enumerate(zip(d["_key"], d["reference_date"])):
        prev = last_seen.get(key)
        if prev is not None and (when - prev[1]).days <= REPOST_WINDOW_DAYS:
            run_id.append(prev[0])
        else:
            run_id.append(i)
        last_seen[key] = (run_id[-1], when)
    d["_run"] = run_id
    agg = d.groupby("_run").agg(times_published=("reference_date", "size"), last_published_at=("reference_date", "max"))
    reposts = d["_run"] != d.index
    note("Повторные публикации объединены", f"тот же источник, заголовок, работодатель и зарплата в течение {REPOST_WINDOW_DAYS} дней; "
         "оставлена первая публикация, число публикаций в times_published", reposts.sum())
    d = d[~reposts].join(agg, on="_run")

    # ---------- derived columns
    d["year"] = d["reference_date"].dt.year
    d["month"] = d["reference_date"].dt.to_period("M").astype(str)
    full = (d["job_title"].fillna("") + " " + d["ad_text"].fillna("")).str.lower()
    d["level_in_title"] = d["job_title"].map(level_in_title)
    d["work_format"] = [work_format_rule(t, x) for t, x in zip(d["work_format_raw"], full)]
    d["education_required"] = education_rule(full)
    d["gender_required"] = gender_rule(full)
    d["schedule"] = schedule_rule(full)
    d["commission_or_bonus"] = commission_rule(full)
    d["city_group"] = d["locality"].map(CITY_GROUP)
    d = salary_columns(d, d["period_text"])
    d["source"] = d["source_key"].map(SOURCE_NAME)
    for s in ["employer", "job_title", "locality", "additional_info", "skills", "address"]:
        d[s] = d[s].replace({"": None})
    for s in ["ad_text", "additional_info", "salary_text"]:
        d[s] = d[s].map(mask_phones)

    cols = ["vacancy_id", "source", "url",
            "reference_date", "date_basis", "published_at", "last_updated_at", "times_published", "last_published_at",
            "year", "month",
            "job_title", "employer", "sector", "profession", "profession_basis", "level_in_title", "level_in_post",
            "experience_required", "experience_min_years", "experience_basis",
            "work_format", "work_format_raw", "education_required", "gender_required", "schedule",
            "city_group", "locality", "address",
            "salary_status", "salary_text", "salary_from", "salary_to", "currency", "salary_basis",
            "pay_period", "pay_period_basis", "fx_rate", "fx_rate_date", "salary_month_kgs", "commission_or_bonus",
            "skills", "additional_info", "ad_text"]
    d = d[cols].sort_values(["reference_date", "vacancy_id"]).reset_index(drop=True)
    d["experience_min_years"] = d["experience_min_years"].astype("Int64")
    d["times_published"] = d["times_published"].astype(int)
    return d, pd.DataFrame(LOG), acc


DICTIONARY = [
    ("vacancy_id", "идентификатор: hh_…, lalafo_…, tg_codifynews_<пост>, tg_kgworkzone_<пост>_<номер вакансии в посте>"),
    ("source", "hh.kg / lalafo.kg / Telegram @codifynews / Telegram @kgworkzone"),
    ("url", "ссылка на объявление или пост"),
    ("reference_date", "ДАТА ДЛЯ АНАЛИЗА (время Бишкека): hh и Telegram: публикация; lalafo: последнее обновление объявления"),
    ("date_basis", "откуда взята reference_date"),
    ("published_at", "первая публикация (время Бишкека, UTC+6)"),
    ("last_updated_at", "последнее изменение объявления (только hh и lalafo)"),
    ("times_published", "сколько раз вакансия была опубликована в течение 30 дней (повторы объединены в эту строку)"),
    ("last_published_at", "дата последней из этих публикаций"),
    ("year", "год reference_date"),
    ("month", "месяц reference_date"),
    ("job_title", "название вакансии как в объявлении"),
    ("employer", "работодатель; пусто, если не указан (почти всегда на lalafo и в @kgworkzone)"),
    ("sector", "сфера (27 значений)"),
    ("profession", "профессия (около 170 значений)"),
    ("profession_basis", "указана источником (hh, lalafo) / классификатор по заголовку (Telegram)"),
    ("level_in_title", "уровень из заголовка: junior / middle / senior / lead; пусто, если не написан"),
    ("level_in_post", "уровень из поля «Тип» в @codifynews (intern, junior, middle, senior, lead); в опыт НЕ переводится"),
    ("experience_required", "без опыта / 1–3 года / 3–6 лет / 6+ лет / требуется, срок не указан; пусто, если не указан"),
    ("experience_min_years", "нижняя граница опыта: 0, 1, 3, 6; пусто, если срок не указан"),
    ("experience_basis", "поле сайта / поле «Дополнительно» / текст объявления"),
    ("work_format", "офис, полный день / удалённо или гибрид / частичная занятость / проектная работа / стажировка; пусто, если не указан"),
    ("work_format_raw", "поле «Тип» поста @codifynews как есть"),
    ("education_required", "высшее / среднее специальное, если упомянуто в тексте"),
    ("gender_required", "женщины / мужчины / женщины или мужчины, если упомянуто в тексте"),
    ("schedule", "график из текста (5/2, 2/2 …)"),
    ("city_group", "Бишкек / Ош / Регионы; пусто, если город не указан"),
    ("locality", "населённый пункт как указан в объявлении"),
    ("address", "адрес (только hh)"),
    ("salary_status", "указана / скрыта / неоплачиваемая / в рублях или тенге, не пересчитана / неправдоподобная"),
    ("salary_text", "строка зарплаты как в посте (Telegram)"),
    ("salary_from", "зарплата «от» в валюте и периоде объявления"),
    ("salary_to", "зарплата «до» в валюте и периоде объявления"),
    ("currency", "KGS / USD / EUR / RUB / KZT"),
    ("salary_basis", "середина диапазона / нижняя граница / верхняя граница"),
    ("pay_period", "месяц / день / неделя / час"),
    ("pay_period_basis", "указано в тексте / по сумме / не указано, считается месячной"),
    ("fx_rate", "курс НБКР для USD и EUR (сом за единицу)"),
    ("fx_rate_date", "дата курса"),
    ("salary_month_kgs", "ИТОГОВАЯ ЗАРПЛАТА: сом в месяц; заполнена только при salary_status = «указана»"),
    ("commission_or_bonus", "в оплате упомянуты проценты или бонусы"),
    ("skills", "ключевые навыки (почти только hh)"),
    ("additional_info", "всё содержание объявления, не вынесенное в колонки (условия, график, требования, контакты скрыты)"),
    ("ad_text", "полный текст объявления или поста"),
]

RULES = [
    ("Даты", "Все даты переведены в время Бишкека (UTC+6). Дата для анализа (reference_date): hh и Telegram: дата публикации; "
             "lalafo: дата последнего обновления (объявления поднимают, и в снимке 2024 года были живые объявления, впервые "
             "опубликованные в 2017–2023)."),
    ("Период", "В таблицу входят вакансии с reference_date с 1 января 2024. Объявления lalafo, не обновлявшиеся с 2023, удалены."),
    ("Повторы", "Одинаковые источник, заголовок, работодатель и строка зарплаты (для lalafo и kgworkzone также начало текста) "
                "в пределах 30 дней = одна вакансия: остаётся первая публикация, число публикаций в times_published. "
                "Повтор через 30+ дней считается новым набором."),
    ("Не вакансии", "lalafo: резюме («ищу работу») и услуги/аренда удалены. Telegram: посты codifynews не по шаблону и посты "
                    "kgworkzone без признаков вакансии удалены."),
    ("Опыт", "Только явно указанный: поле сайта (hh, lalafo), поле «Дополнительно» (codifynews), фраза в тексте (kgworkzone). "
             "«С опытом / опыт обязателен» без срока = «требуется, срок не указан». Уровень Middle/Senior НЕ переводится в годы. "
             "Если не указан, пусто."),
    ("Город", "Только явно указанный. Название канала («Работа Бишкек») городом не считается."),
    ("Профессия", "hh и lalafo: из источника (с исправлением ошибки справочника для менеджеров по продажам). Telegram: "
                  "классификатор по заголовку (TF-IDF + LinearSVC), обученный на hh и lalafo."),
    ("Зарплата", "Диапазон: середина; одна граница: она сама. Валюта не указана (lalafo): KGS. USD и EUR по курсу НБКР своего "
                 "года (2024: 14.03.2024; 2025: 02.07.2025; 2026: 30.06.2026). RUB и KZT не пересчитываются: строка остаётся, "
                 "зарплата помечена."),
    ("Период оплаты", "Сначала явные слова («в час», «за смену/в день», «в месяц», «в неделю»); «в месяц» важнее «выплаты раз "
                      "в неделю». Без слов: 500–5 000 сом считаются дневной ставкой, остальное месячной. Пересчёт: час × 208, "
                      "день × 26, неделя × 4,33."),
    ("Неправдоподобные зарплаты", "Меньше 10 тыс. или больше 600 тыс. сом в месяц после пересчёта: зарплата не используется, "
                                  "вакансия остаётся со статусом."),
    ("Текст", "Полный текст в ad_text; всё, что не вынесено в отдельные колонки, в additional_info. Телефоны и @username "
              "скрыты при сборе."),
]
