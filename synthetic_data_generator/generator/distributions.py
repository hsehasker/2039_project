"""Feature distribution generators for synthetic data.

Supports normal, lognormal, uniform, exponential, categorical,
bernoulli, sequential_id, template_email, template_phone,
date_from_age, and random_date_range generators.
All parameters are configurable via YAML config.
"""

from datetime import date, timedelta
from typing import Any

import numpy as np
from numpy.random import Generator

LAST_NAMES_MALE: list[str] = [
    "Иванов", "Петров", "Сидоров", "Козлов", "Новиков",
    "Морозов", "Волков", "Соловьёв", "Васильев", "Зайцев",
    "Павлов", "Семёнов", "Голубев", "Виноградов", "Богданов",
    "Воробьёв", "Фёдоров", "Михайлов", "Беляев", "Тарасов",
    "Белов", "Комаров", "Орлов", "Киселёв", "Макаров",
    "Андреев", "Ковалёв", "Ильин", "Гусев", "Титов",
    "Кузьмин", "Кудрявцев", "Баранов", "Куликов", "Алексеев",
    "Степанов", "Яковлев", "Сорокин", "Сергеев", "Романов",
    "Захаров", "Борисов", "Королёв", "Герасимов", "Пономарёв",
    "Григорьев", "Лазарев", "Медведев", "Ершов", "Никитин",
]
LAST_NAMES_FEMALE: list[str] = [name + "а" for name in LAST_NAMES_MALE]

FIRST_NAMES_MALE: list[str] = [
    "Александр", "Дмитрий", "Максим", "Иван", "Артём",
    "Сергей", "Андрей", "Алексей", "Михаил", "Николай",
    "Кирилл", "Даниил", "Егор", "Виктор", "Роман",
    "Владимир", "Павел", "Олег", "Тимофей", "Денис",
]
FIRST_NAMES_FEMALE: list[str] = [
    "Анна", "Мария", "Елена", "Ольга", "Наталья",
    "Екатерина", "Татьяна", "Людмила", "Светлана", "Ирина",
    "Юлия", "Дарья", "Алина", "Виктория", "Полина",
    "Ксения", "Валерия", "Софья", "Вера", "Маргарита",
]

MIDDLE_NAMES_MALE: list[str] = [
    "Александрович", "Дмитриевич", "Максимович", "Иванович", "Сергеевич",
    "Андреевич", "Алексеевич", "Михайлович", "Николаевич", "Владимирович",
    "Павлович", "Олегович", "Викторович", "Петрович", "Романович",
]
MIDDLE_NAMES_FEMALE: list[str] = [
    "Александровна", "Дмитриевна", "Максимовна", "Ивановна", "Сергеевна",
    "Андреевна", "Алексеевна", "Михайловна", "Николаевна", "Владимировна",
    "Павловна", "Олеговна", "Викторовна", "Петровна", "Романовна",
]


class FeatureGenerator:
    def __init__(self, rng: Generator) -> None:
        self.rng = rng
        self.genders = None  # None initially, will be generated once

    def _get_genders(self, size):
        if self.genders is None or len(self.genders) != size:
            self.genders = self.rng.integers(0, 2, size=size) # 0 for male, 1 for female
        return self.genders

    def normal(self, mu, sigma, size, clip=None):
        values = self.rng.normal(loc=mu, scale=sigma, size=size)
        return _apply_clip(values, clip)

    def lognormal(self, mu, sigma, size):
        values = self.rng.lognormal(mean=mu, sigma=sigma, size=size)
        return np.round(values, 2)

    def uniform_int(self, low, high, size):
        return self.rng.integers(low=low, high=high + 1, size=size)

    def exponential(self, lam, size, clip=None):
        scale = 1.0 / lam
        values = self.rng.exponential(scale=scale, size=size)
        return _apply_clip(values, clip)

    def categorical(self, categories, size):
        labels = list(categories.keys())
        probs = np.array(list(categories.values()), dtype=float)
        probs /= probs.sum()
        return self.rng.choice(labels, size=size, p=probs)

    def bernoulli(self, p, size):
        return self.rng.random(size=size) < p

    def sequential_id(self, prefix, size):
        return np.array([f"{prefix}-{i:06d}" for i in range(1, size + 1)])

    def date_from_age(self, age_array, reference_date):
        ref = date.fromisoformat(reference_date)
        dates = []
        for age in age_array:
            age_int = int(age)
            birth_year = ref.year - age_int
            day_offset = int(self.rng.integers(0, 365))
            try:
                birth = date(birth_year, ref.month, ref.day) - timedelta(days=day_offset)
            except ValueError:
                birth = date(birth_year, ref.month, 28) - timedelta(days=day_offset)
            dates.append(birth.isoformat())
        return np.array(dates)

    def random_date_range(self, start, end, size):
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        total_days = (end_date - start_date).days
        offsets = self.rng.integers(0, total_days, size=size)
        return np.array([(start_date + timedelta(days=int(d))).isoformat() for d in offsets])

    def template_email(self, n_array, domains):
        import string
        domain_names = list(domains.keys())
        domain_probs = np.array(list(domains.values()), dtype=float)
        domain_probs /= domain_probs.sum()
        
        size = len(n_array)
        chosen_domains = self.rng.choice(domain_names, size=size, p=domain_probs)
        
        chars = list(string.ascii_letters + string.digits + "." + "_")
        usernames = set()
        
        while len(usernames) < size:
            # Generate more than needed to account for duplicates, or just loop
            needed = size - len(usernames)
            lengths = self.rng.integers(6, 16, size=needed)
            for length in lengths:
                username = "".join(self.rng.choice(chars, size=length))
                usernames.add(username)
                if len(usernames) >= size:
                    break
                    
        usernames_list = list(usernames)
        return np.array([f"{u}@{d}" for u, d in zip(usernames_list, chosen_domains)])

    def template_phone(self, size):
        phones = []
        for _ in range(size):
            digits = self.rng.integers(0, 10, size=10)
            phone = (
                f"+7-9{digits[0]}{digits[1]}"
                f"-{digits[2]}{digits[3]}{digits[4]}"
                f"-{digits[5]}{digits[6]}"
                f"-{digits[7]}{digits[8]}"
            )
            phones.append(phone)
        return np.array(phones)

    def generate_last_name(self, size):
        genders = self._get_genders(size)
        return np.array([self.rng.choice(LAST_NAMES_MALE) if g == 0 else self.rng.choice(LAST_NAMES_FEMALE) for g in genders])

    def generate_first_name(self, size):
        genders = self._get_genders(size)
        return np.array([self.rng.choice(FIRST_NAMES_MALE) if g == 0 else self.rng.choice(FIRST_NAMES_FEMALE) for g in genders])

    def generate_middle_name(self, size):
        genders = self._get_genders(size)
        return np.array([self.rng.choice(MIDDLE_NAMES_MALE) if g == 0 else self.rng.choice(MIDDLE_NAMES_FEMALE) for g in genders])


def generate_feature(
    name: str,
    config: dict[str, Any],
    n_samples: int,
    rng: Generator,
    reference_date: str = "2025-01-01",
    age_array: np.ndarray | None = None,
    id_array: np.ndarray | None = None,
    gen: FeatureGenerator | None = None,
) -> np.ndarray:
    if gen is None:
        gen = FeatureGenerator(rng)

    feat_type = config.get("type")
    if feat_type:
        if feat_type == "sequential_id":
            prefix = config.get("prefix", "USR")
            return gen.sequential_id(prefix, n_samples)
        elif feat_type == "categorical":
            if name == "last_name":
                return gen.generate_last_name(n_samples)
            elif name == "first_name":
                return gen.generate_first_name(n_samples)
            elif name == "middle_name":
                return gen.generate_middle_name(n_samples)
        elif feat_type == "derived_from_age":
            if age_array is None:
                raise ValueError("age_array is required for derived_from_age")
            return gen.date_from_age(age_array, reference_date)
        elif feat_type == "template_email":
            domains = config.get("domains", {"mail.ru": 0.4, "gmail.com": 0.3, "yandex.ru": 0.3})
            if id_array is None:
                id_array = np.arange(1, n_samples + 1)
            return gen.template_email(id_array, domains)
        elif feat_type == "template_phone":
            return gen.template_phone(n_samples)
        elif feat_type == "random_date_range":
            start = config.get("start", "2020-01-01")
            end = config.get("end", "2025-01-01")
            return gen.random_date_range(start, end, n_samples)
        else:
            raise ValueError(f"Unknown type '{feat_type}' for feature '{name}'")

    dist = config.get("distribution")
    if dist is None:
        raise ValueError(f"No 'distribution' or 'type' specified for feature '{name}'")

    if dist == "normal":
        clip = config.get("clip")
        return gen.normal(config["mean"], config["std"], n_samples, clip)
    elif dist == "lognormal":
        return gen.lognormal(config["mean"], config["std"], n_samples)
    elif dist == "exponential":
        lam = config["lambda"]
        clip = config.get("clip")
        return gen.exponential(lam, n_samples, clip)
    elif dist == "categorical":
        return gen.categorical(config["categories"], n_samples)
    elif dist == "bernoulli":
        return gen.bernoulli(config["p"], n_samples)
    else:
        raise ValueError(f"Unknown distribution '{dist}' for feature '{name}'")


def _apply_clip(values, clip):
    if clip is None:
        return values
    low, high = clip
    if low is not None:
        values = np.maximum(values, low)
    if high is not None:
        values = np.minimum(values, high)
    return values
