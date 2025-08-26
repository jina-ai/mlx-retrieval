#!/usr/bin/env python3

from faker import Faker

fake = Faker(
    [
        "zh_CN",
        "en_US",
        "zh_TW",
        "ja_JP",
        "ko_KR",
        "ru_RU",
        "ar_SA",
        "de_DE",
        "es_ES",
        "fr_FR",
        "it_IT",
        "pt_BR",
        "tr_TR",
        "vi_VN",
        "ar_AE",
    ]
)
fake.seed_instance(4321)

all_data = set()
for _ in range(5000):
    data = [
        fake.address(),
        fake.name(),
        fake.phone_number(),
        fake.email(),
        fake.city(),
        fake.building_number(),
        fake.street_name(),
        fake.street_address(),
        fake.secondary_address(),
        fake.postcode(),
        fake.state(),
        fake.country(),
        fake.job(),
        fake.company(),
        fake.sentence(nb_words=10),
        fake.sentence(nb_words=20),
        fake.sentence(nb_words=40),
        fake.sentence(nb_words=60),
        fake.sentence(nb_words=120),
        fake.sentence(nb_words=256),
        fake.sentence(nb_words=512),
    ]
    for item in data:
        if item and item.strip():
            all_data.add(item.strip())

with open("data/v8.txt", "w") as f:
    for line in sorted(all_data):
        f.write(line + "\n")
