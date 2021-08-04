from fastapi import FastAPI
from typing import Optional, List, Dict
from pydantic import BaseModel
from ML.Extractive.textrank import extractive_textrank
from ML.Extractive.luhn import extractive_luhn
from ML.Extractive.lsa import extractive_lsa
from ML.Extractive.lexrank import extractive_lexrank

app = FastAPI()

class Text(BaseModel):
    text: str
    len: int
    



@app.get("/")
def home():
    return {"hello"}

@app.post("/summarize/extractive/textrank")
def summarize_textrank(text: Text):
    return {"text_summary": extractive_textrank(text.text, text.len) }

@app.post("/summarize/extractive/lexrank")
def summarize_lexrank(text: Text):
    return {"text_summary": extractive_lexrank(text.text, text.len) }

@app.post("/summarize/extractive/luhn")
def summarize_luhn(text: Text):
    return {"text_summary": extractive_luhn(text.text, text.len) }

@app.post("/summarize/extractive/lsa")
def summarize_lsa(text: Text):
    return {"text_summary": extractive_lsa(text.text, text.len) }