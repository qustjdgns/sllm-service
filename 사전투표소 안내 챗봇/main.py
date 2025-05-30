import json
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import uvicorn
from difflib import SequenceMatcher

# HuggingFace 토큰 환경변수 설정 (본인 토큰으로 교체)
os.environ['HF_TOKEN'] = ""

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

print("모델과 토크나이저 불러오는 중...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=os.environ['HF_TOKEN']
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=os.environ['HF_TOKEN']
)

print("파이프라인 생성 중...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 모든 도메인 허용, 운영 시 수정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSON 데이터 파일 경로 및 로드 (utf-8 인코딩)
with open("전국사전투표소.json", encoding="utf-8") as f:
    polling_places = json.load(f)

class Query(BaseModel):
    question: str

def normalize_location(loc: str) -> str:
    loc = loc.replace(" ", "").lower()

    abbreviations = {
        "서울시": "서울특별시",
        "부산시": "부산광역시",
        "대구시": "대구광역시",
        "인천시": "인천광역시",
        "광주시": "광주광역시",
        "대전시": "대전광역시",
        "울산시": "울산광역시",
        "세종시": "세종특별자치시",
        "경기": "경기도",
        "강원": "강원도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전라북도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주특별자치도",
    }

    for abbr, full in abbreviations.items():
        if loc.startswith(abbr):
            loc = full + loc[len(abbr):]
            break

    return loc

def safe_concat_parts(sd: str, wiw: str, emd: str) -> str:
    sd = sd or ""
    wiw = wiw or ""
    emd = emd or ""

    # 중복되는 접미사 제거 (필요시 더 정교화 가능)
    if sd and wiw and wiw.startswith(sd[-1]):
        wiw = wiw[1:]
    if wiw and emd and emd.startswith(wiw[-1]):
        emd = emd[1:]

    return sd + wiw + emd

def extract_location_from_question(question: str) -> str:
    prompt = (
        f"사용자의 입력: '{question}'\n"
        "이 문장에서 사전투표소 검색에 필요한 행정지역명(시/군/구, 읍/면/동)을 띄어쓰기를 포함해 올바르게 한 줄로 추출해줘.\n"
        "예: '성남시 수정구 산성동'\n"
        "답변:"
    )
    result = generator(prompt, max_new_tokens=32, do_sample=False)[0]["generated_text"]
    lines = result.strip().split("\n")
    answer_line = lines[-1]
    if "답변:" in answer_line:
        answer_line = answer_line.split("답변:")[-1].strip()

    # 여러 공백을 하나로 정리, 특수문자 제거
    answer_line = re.sub(r"\s+", " ", answer_line).strip()

    return answer_line

def normalize_text(text):
    text = re.sub(r"\s+", "", text)  # 공백 제거
    text = ''.join(e for e in text if e.isalnum())  # 특수문자 제거
    return text.lower()

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def find_polling_place(location: str, min_threshold: float = 0.8):
    location_norm = normalize_text(location)
    best_match = None
    best_score = 0

    for place in polling_places:
        sd = place.get('sdName', '')
        wiw = place.get('wiwName', '')
        emd = place.get('emdName', '')

        candidates = [
            normalize_text(f"{sd} {wiw} {emd}"),
            normalize_text(f"{sd} {emd}"),
            normalize_text(f"{wiw} {emd}"),
            normalize_text(sd),
            normalize_text(emd),
        ]

        for candidate in candidates:
            sim = similarity(location_norm, candidate)
            print(f"Comparing '{location_norm}' vs '{candidate}' => Similarity: {sim:.2f}")

            if sim > best_score and sim >= min_threshold:
                best_score = sim
                best_match = place

    if best_match:
        return [best_match]
    else:
        return []


def generate_answer(place: dict, user_question: str, location_extracted: str):
    place_info = (
        f"주소: {place.get('addr', '정보 없음')}\n"
        f"장소명: {place.get('placeName', '정보 없음')}\n"
        f"층수: {place.get('floor', '정보 없음')}\n"
    )
    prompt = (
        f"사용자의 질문: '{user_question}'\n"
        f"추출된 위치명: '{location_extracted}'\n"
        f"다음은 해당 지역의 사전투표소 정보입니다:\n"
        f"{place_info}\n"
        "위 정보를 기반으로 자연스럽고 친절한 응답 문장을 생성해주세요.\n답변:"
    )
    result = generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    lines = result.strip().split("\n")
    answer_line = lines[-1]
    if "답변:" in answer_line:
        answer_line = answer_line.split("답변:")[-1].strip()
    return answer_line

@app.post("/api/ask")
async def ask_polling_place(query: Query):
    location = extract_location_from_question(query.question)
    print(f"[사용자 질문] {query.question}")
    print(f"[추출된 위치명] '{location}'")

    if not location:
        return {"answer": "죄송합니다. 정확한 지역 정보를 추출할 수 없었습니다. 시/군/구와 읍/면/동을 포함해 다시 말씀해 주세요."}

    matched_places = find_polling_place(location)
    if not matched_places:
        return {"answer": f"죄송합니다, '{location}' 지역의 사전투표소 정보를 찾을 수 없습니다."}

    place = matched_places[0]
    answer = generate_answer(place, query.question, location)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "EXAONE 기반 사전투표소 안내 챗봇 실행 중"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


