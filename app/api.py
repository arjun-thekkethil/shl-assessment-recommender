"""FastAPI service — /health, /chat, and chat UI at /."""
from __future__ import annotations

import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .agent import SHLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_api")

app = FastAPI(title="SHL Assessment Recommender", docs_url="/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: SHLAgent | None = None

# ---------------------------------------------------------------------------
# Chat UI — served inline so no static file copy is needed in Docker
# ---------------------------------------------------------------------------
_UI = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SHL Assessment Recommender</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:#0a0f1e; --surface:#111827; --surface2:#1a2235;
      --border:rgba(255,255,255,0.08); --accent:#6366f1; --accent2:#818cf8;
      --text:#e2e8f0; --muted:#64748b; --user-bg:#1e3a5f; --bot-bg:#1a2235; --radius:14px;
    }
    html,body{height:100%;font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);}
    .shell{display:grid;grid-template-columns:1fr 380px;grid-template-rows:56px 1fr;height:100vh;max-width:1280px;margin:0 auto;}
    .topbar{grid-column:1/-1;display:flex;align-items:center;gap:12px;padding:0 24px;border-bottom:1px solid var(--border);background:var(--surface);}
    .topbar-logo{font-size:1.1rem;font-weight:700;letter-spacing:.02em;}
    .topbar-logo span{color:var(--accent2);}
    .topbar-badge{font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;border:1px solid var(--border);border-radius:999px;padding:2px 8px;color:var(--muted);}
    .topbar-spacer{flex:1;}
    .btn-new{font-size:.8rem;padding:6px 14px;border-radius:999px;border:1px solid var(--border);background:transparent;color:var(--text);cursor:pointer;}
    .btn-new:hover{background:var(--surface2);}
    .chat-col{display:flex;flex-direction:column;border-right:1px solid var(--border);overflow:hidden;}
    .messages{flex:1;overflow-y:auto;padding:24px 20px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth;}
    .welcome{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;text-align:center;color:var(--muted);padding:40px;}
    .welcome-icon{font-size:2.4rem;}
    .welcome h2{font-size:1.1rem;color:var(--text);}
    .welcome p{font-size:.85rem;max-width:340px;line-height:1.5;}
    .welcome-chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:8px;}
    .chip{font-size:.78rem;padding:6px 14px;border-radius:999px;border:1px solid var(--border);color:var(--text);cursor:pointer;background:var(--surface2);}
    .chip:hover{border-color:var(--accent2);color:var(--accent2);}
    .msg{display:flex;gap:10px;max-width:85%;}
    .msg.user{align-self:flex-end;flex-direction:row-reverse;}
    .msg.assistant{align-self:flex-start;}
    .avatar{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.85rem;}
    .msg.user .avatar{background:var(--accent);}
    .msg.assistant .avatar{background:var(--surface2);border:1px solid var(--border);}
    .bubble{border-radius:var(--radius);padding:10px 14px;font-size:.88rem;line-height:1.55;}
    .msg.user .bubble{background:var(--user-bg);border-bottom-right-radius:4px;}
    .msg.assistant .bubble{background:var(--bot-bg);border:1px solid var(--border);border-bottom-left-radius:4px;}
    .typing{display:flex;gap:5px;padding:4px 2px;}
    .typing span{width:7px;height:7px;border-radius:50%;background:var(--muted);animation:bounce .9s infinite;}
    .typing span:nth-child(2){animation-delay:.15s;}
    .typing span:nth-child(3){animation-delay:.3s;}
    @keyframes bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-6px);}}
    .input-bar{padding:14px 16px;border-top:1px solid var(--border);display:flex;gap:10px;align-items:flex-end;background:var(--surface);}
    .input-bar textarea{flex:1;resize:none;border-radius:12px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font:inherit;font-size:.9rem;padding:10px 14px;max-height:140px;outline:none;line-height:1.5;}
    .input-bar textarea:focus{border-color:var(--accent);}
    .input-bar textarea::placeholder{color:var(--muted);}
    .send-btn{width:40px;height:40px;border-radius:50%;border:none;background:var(--accent);color:#fff;cursor:pointer;font-size:1rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
    .send-btn:hover:not(:disabled){background:var(--accent2);}
    .send-btn:disabled{opacity:.45;cursor:default;}
    .rec-col{display:flex;flex-direction:column;overflow:hidden;background:var(--surface);}
    .rec-header{padding:14px 16px 10px;border-bottom:1px solid var(--border);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);display:flex;align-items:center;justify-content:space-between;}
    .rec-count{background:var(--surface2);border-radius:999px;padding:2px 9px;font-size:.72rem;color:var(--accent2);}
    .rec-list{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;}
    .rec-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:10px;color:var(--muted);text-align:center;padding:32px;}
    .rec-empty-icon{font-size:2rem;}
    .rec-empty p{font-size:.82rem;}
    .rec-card{border-radius:10px;border:1px solid var(--border);background:var(--surface2);padding:10px 12px;transition:border-color .15s;}
    .rec-card:hover{border-color:var(--accent);}
    .rec-name{font-size:.85rem;font-weight:600;margin-bottom:6px;line-height:1.3;}
    .rec-name a{color:var(--text);text-decoration:none;}
    .rec-name a:hover{color:var(--accent2);}
    .rec-meta{display:flex;gap:6px;flex-wrap:wrap;}
    .badge{font-size:.68rem;padding:2px 8px;border-radius:999px;border:1px solid var(--border);color:var(--muted);}
    .badge-type{border-color:var(--accent);color:var(--accent2);}
    .legend{padding:10px 12px;border-top:1px solid var(--border);font-size:.68rem;color:var(--muted);line-height:1.8;}
    .legend strong{color:var(--text);}
    ::-webkit-scrollbar{width:5px;}
    ::-webkit-scrollbar-track{background:transparent;}
    ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:4px;}
    @media(max-width:768px){
      .shell{grid-template-columns:1fr;grid-template-rows:56px 1fr auto;}
      .rec-col{max-height:260px;border-right:none;border-top:1px solid var(--border);}
    }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="topbar-logo">SHL <span>Assess</span></div>
    <span class="topbar-badge">Recommender</span>
    <div class="topbar-spacer"></div>
    <button class="btn-new" id="btn-new">+ New chat</button>
  </header>
  <main class="chat-col">
    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <div class="welcome-icon">&#127919;</div>
        <h2>SHL Assessment Recommender</h2>
        <p>Describe a role or hiring need and I&#39;ll suggest the most relevant SHL assessments from the catalog.</p>
        <div class="welcome-chips">
          <span class="chip" data-q="I need to hire a Java backend developer">Java developer</span>
          <span class="chip" data-q="Sales representative with strong communication skills">Sales rep</span>
          <span class="chip" data-q="Entry-level data analyst for a financial firm">Data analyst</span>
          <span class="chip" data-q="Senior software engineer, 8+ years, leadership skills">Senior engineer</span>
        </div>
      </div>
    </div>
    <div class="input-bar">
      <textarea id="input" rows="1" placeholder="Describe a role or hiring need&#8230;"></textarea>
      <button class="send-btn" id="send-btn" disabled title="Send">&#9658;</button>
    </div>
  </main>
  <aside class="rec-col">
    <div class="rec-header">Recommendations<span class="rec-count" id="rec-count">0</span></div>
    <div class="rec-list" id="rec-list">
      <div class="rec-empty"><div class="rec-empty-icon">&#128203;</div><p>Recommended assessments will appear here after you describe a role.</p></div>
    </div>
    <div class="legend"><strong>Types:</strong> A=Ability &nbsp;B=Biodata &nbsp;C=Competency<br>D=Development &nbsp;E=Engagement &nbsp;K=Knowledge<br>P=Personality &nbsp;S=Situational</div>
  </aside>
</div>
<script>
const API="";
let history=[],busy=false;
const messagesEl=document.getElementById("messages"),inputEl=document.getElementById("input"),
      sendBtn=document.getElementById("send-btn"),recList=document.getElementById("rec-list"),
      recCount=document.getElementById("rec-count");
let welcomeEl=document.getElementById("welcome");
inputEl.addEventListener("input",()=>{inputEl.style.height="auto";inputEl.style.height=Math.min(inputEl.scrollHeight,140)+"px";sendBtn.disabled=busy||!inputEl.value.trim();});
inputEl.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage();}});
sendBtn.addEventListener("click",sendMessage);
document.getElementById("btn-new").addEventListener("click",resetChat);
document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{inputEl.value=c.dataset.q;inputEl.dispatchEvent(new Event("input"));sendMessage();}));
function scrollBottom(){messagesEl.scrollTop=messagesEl.scrollHeight;}
function addBubble(role,text){
  if(welcomeEl){welcomeEl.remove();welcomeEl=null;}
  const d=document.createElement("div");d.className="msg "+role;
  d.innerHTML=`<div class="avatar">${role==="user"?"&#128100;":"&#129302;"}</div><div class="bubble">${esc(text)}</div>`;
  messagesEl.appendChild(d);scrollBottom();return d;
}
function addTyping(){
  if(welcomeEl){welcomeEl.remove();welcomeEl=null;}
  const d=document.createElement("div");d.className="msg assistant";d.id="typing";
  d.innerHTML=`<div class="avatar">&#129302;</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  messagesEl.appendChild(d);scrollBottom();return d;
}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\n/g,"<br>");}
const TYPES={A:"Ability",B:"Biodata",C:"Competency",D:"Development",E:"Engagement",K:"Knowledge",P:"Personality",S:"Situational"};
function renderRecs(recs){
  recCount.textContent=recs.length;
  if(!recs.length)return;
  recList.innerHTML=recs.map((r,i)=>`<div class="rec-card"><div class="rec-name"><span style="color:var(--muted);font-size:.75rem;">${i+1}.</span> <a href="${r.url}" target="_blank" rel="noopener">${esc(r.name)}</a></div><div class="rec-meta"><span class="badge badge-type">${r.test_type} &middot; ${TYPES[r.test_type]||r.test_type}</span></div></div>`).join("");
}
async function sendMessage(){
  const text=inputEl.value.trim();if(!text||busy)return;
  busy=true;sendBtn.disabled=true;inputEl.value="";inputEl.style.height="auto";
  addBubble("user",text);history.push({role:"user",content:text});
  const t=addTyping();
  try{
    const res=await fetch(API+"/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({messages:history})});
    if(!res.ok)throw new Error("HTTP "+res.status);
    const data=await res.json();
    t.remove();addBubble("assistant",data.reply||"...");
    history.push({role:"assistant",content:data.reply||""});
    if(data.recommendations?.length)renderRecs(data.recommendations);
  }catch(e){t.remove();addBubble("assistant","Sorry, couldn't reach the server. Please try again.");console.error(e);}
  finally{busy=false;sendBtn.disabled=!inputEl.value.trim();}
}
function resetChat(){
  history=[];
  messagesEl.innerHTML=`<div class="welcome" id="welcome"><div class="welcome-icon">&#127919;</div><h2>SHL Assessment Recommender</h2><p>Describe a role or hiring need and I'll suggest the most relevant SHL assessments.</p><div class="welcome-chips"><span class="chip" data-q="I need to hire a Java backend developer">Java developer</span><span class="chip" data-q="Sales representative with strong communication skills">Sales rep</span><span class="chip" data-q="Entry-level data analyst for a financial firm">Data analyst</span><span class="chip" data-q="Senior software engineer, 8+ years, leadership skills">Senior engineer</span></div></div>`;
  welcomeEl=document.getElementById("welcome");
  document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{inputEl.value=c.dataset.q;inputEl.dispatchEvent(new Event("input"));sendMessage();}));
  recList.innerHTML=`<div class="rec-empty"><div class="rec-empty-icon">&#128203;</div><p>Recommendations appear here.</p></div>`;
  recCount.textContent="0";
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    global _agent
    logger.info("Initializing SHLAgent …")
    _agent = SHLAgent()
    logger.info("SHLAgent ready")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "HEAD"])
def serve_ui(request: Request):
    if request.method == "HEAD":
        return Response(headers={"content-type": "text/html; charset=utf-8"})
    return HTMLResponse(content=_UI)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = _agent.chat(messages)

    recs = [
        Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result.get("end_of_conversation", False),
    )
