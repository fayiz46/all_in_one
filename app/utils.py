# app/utils.py
import re


def validate_aadhaar(number: str) -> bool:
    return bool(re.fullmatch(r"\d{4} \d{4} \d{4}", number))

def validate_pan(pan: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))

def validate_dl(dl: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}\d{2} \d{11}", dl))
