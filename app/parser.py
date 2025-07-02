# app/parser.py

import pytesseract
import cv2
import numpy as np
import re
from app.utils import validate_aadhaar, validate_pan, validate_dl

def parse_document(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)

    result = {
        "document_type": "Unknown",
        "fields": {},
        "raw_text": text
    }

    # Aadhaar
    aadhaar = re.search(r"\d{4} \d{4} \d{4}", text)
    if aadhaar and validate_aadhaar(aadhaar.group()):
        result["document_type"] = "Aadhaar"
        result["fields"]["aadhaar_number"] = aadhaar.group()
        result["fields"]["dob"] = re.search(r"\d{2}/\d{2}/\d{4}", text).group(0) if re.search(r"\d{2}/\d{2}/\d{4}", text) else ""
        result["fields"]["gender"] = re.search(r"MALE|FEMALE|Male|Female", text).group(0) if re.search(r"MALE|FEMALE|Male|Female", text) else ""
        name_match = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
        result["fields"]["name"] = name_match[0] if name_match else ""

    # PAN
    elif re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
        result["document_type"] = "PAN"
        pan = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text).group(0)
        if validate_pan(pan):
            result["fields"]["pan_number"] = pan
            result["fields"]["name"] = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)[0]
            result["fields"]["father_name"] = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)[1]
            result["fields"]["dob"] = re.search(r"\d{2}/\d{2}/\d{4}", text).group(0)

    # Driving License
    elif re.search(r"[A-Z]{2}\d{2} \d{11}", text):
        result["document_type"] = "Driving License"
        dl = re.search(r"[A-Z]{2}\d{2} \d{11}", text).group(0)
        if validate_dl(dl):
            result["fields"]["dl_number"] = dl
            result["fields"]["dob"] = re.search(r"\d{2}-\d{2}-\d{4}", text).group(0) if re.search(r"\d{2}-\d{2}-\d{4}", text) else ""
            name = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
            result["fields"]["name"] = name[0] if name else ""

    return result
