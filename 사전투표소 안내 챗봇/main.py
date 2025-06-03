import json
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import uvicorn

# HuggingFace 토큰 (필요시 입력)
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

print("텍스트 생성 파이프라인 구성 중...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSON 데이터 불러오기
with open("전국사전투표소.json", encoding="utf-8") as f:
    early_polling_places = json.load(f)

with open("전국투표소.json", encoding="utf-8") as f:
    regular_polling_places = json.load(f)

class Query(BaseModel):
    question: str
    vote_type: str  # "early" or "regular"

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return ''.join(e for e in text if e.isalnum()).lower()

def simple_extract_location(question: str) -> str:
    pattern = re.compile(r'([가-힣]+(?:시|군|구|도))?\s*([가-힣]+(?:동|읍|면))')
    match = pattern.search(question)
    if match:
        groups = [g for g in match.groups() if g]
        unique_groups = []
        for g in groups:
            if g not in unique_groups:
                unique_groups.append(g)
        return ' '.join(unique_groups).strip()
    else:
        return question.strip()

def find_polling_places(location: str, dataset):
    location_norm = normalize_text(location)
    matched_places = []

    for place in dataset:
        full_addr = normalize_text(
            f"{place.get('sdName', '')} {place.get('wiwName', '')} {place.get('emdName', '')}"
        )
        if location_norm in full_addr:
            matched_places.append(place)

    return matched_places

def generate_answer_multiple(places: list, user_question: str, location_extracted: str, vote_type: str):
    if not places:
        return "정보를 찾을 수 없습니다."

    answers = []
    for i, place in enumerate(places, 1):
        answers.append(
            f"[{i}] 장소명: {place.get('placeName', '정보 없음')}\n"
            f"주소: {place.get('addr', '정보 없음')}\n"
            f"층수: {place.get('floor', '정보 없음')}\n"
        )

    combined_info = "\n\n".join(answers)

    # 프롬프트 분기
    if vote_type == "early":
        prompt = (
            f"사용자의 질문: '{user_question}'\n"
            f"추출된 위치명: '{location_extracted}'\n"
            f"다음은 해당 지역의 사전투표소 정보입니다:\n"
            f"{combined_info}\n"
            "위 정보를 바탕으로, 사용자에게 해당 지역 사전투표소의 위치(주소)와 층수 정보만을 "
            "자연스럽고 친절하게 안내하는 문장을 한두 문장 이내로 작성해 주세요. 숫자나 목록 없이 설명 형태로만 답변해 주세요.\n"
            "답변:"
        )
    else:  # regular
        prompt = (
            f"사용자의 질문: '{user_question}'\n"
            f"추출된 위치명: '{location_extracted}'\n"
            f"다음은 해당 지역의 투표소 여러 곳의 정보입니다:\n"
            f"{combined_info}\n"
            "위 정보를 바탕으로, 사용자에게 해당 지역 투표소들의 위치(주소)와 층수 정보만을 "
            "자연스럽고 친절하게 안내하는 문장을 한두 문장 이내로 작성해 주세요. 숫자나 목록 없이 설명 형태로만 답변해 주세요.\n"
            "답변:"
        )

    result = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
    lines = result.strip().split("답변:")[-1].strip().split("\n")
    clean_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(clean_lines)

@app.post("/api/ask")
async def ask_polling_place(query: Query):
    location = simple_extract_location(query.question)
    print(f"[사용자 질문] {query.question}")
    print(f"[추출된 위치명] '{location}'")

    if not location:
        return {"answer": "죄송합니다. 정확한 지역 정보를 추출할 수 없었습니다. 시/군/구와 읍/면/동을 포함해 다시 말씀해 주세요."}

    if query.vote_type == "early":
        dataset = early_polling_places
        vote_type_text = "사전투표소"
    elif query.vote_type == "regular":
        dataset = regular_polling_places
        vote_type_text = "투표소"
    else:
        return {"answer": "알 수 없는 투표소 종류입니다. '사전투표소' 또는 '투표소'를 선택해 주세요."}

    matched_places = find_polling_places(location, dataset)
    if not matched_places:
        return {"answer": f"죄송합니다, '{location}' 지역의 {vote_type_text} 정보를 찾을 수 없습니다."}

    answer = generate_answer_multiple(matched_places, query.question, location, query.vote_type)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "EXAONE 기반 투표소 안내 챗봇 실행 중"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

