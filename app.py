# =============================================================================
#  NutriWise AI – Personalized Nutrition Coach
#  A Multi-Agent AI Application powered by IBM watsonx.ai Granite Models
#
#  Architecture:
#    Agent 1 – Nutrition Knowledge Agent
#    Agent 2 – Diet Planner Agent
#    Agent 3 – Health Advisory Agent
#    Agent 4 – Meal Analysis Agent
#    Orchestrator – Routes requests to the correct agent
#
#  Stack: Python · Flask · Bootstrap 5 · IBM watsonx.ai (ibm-watsonx-ai SDK)
# =============================================================================

import os
import json
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
load_dotenv()

# IBM watsonx.ai SDK
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "nutriwise-secret-2024")

# ---------------------------------------------------------------------------
# IBM watsonx.ai Configuration
# Read credentials from environment variables (never hard-code credentials)
# ---------------------------------------------------------------------------
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# ---------------------------------------------------------------------------
# Core AI Function – calls IBM watsonx.ai Granite model
# All four agents delegate to this single function.
# ---------------------------------------------------------------------------
def generate_response(prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to IBM watsonx.ai (ibm/granite-13b-instruct-v2)
    and return the generated text response.

    Parameters
    ----------
    prompt     : The instruction/question string to send to the model
    max_tokens : Maximum number of tokens the model may generate

    Returns
    -------
    str – The model's text response, or an error message.
    """
    try:
        # ----- Initialise watsonx.ai client -----
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY,
        )
        client = APIClient(credentials=credentials, project_id=WATSONX_PROJECT_ID)

        # ----- Select Granite model -----
        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",   # IBM Granite Instruct
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                "max_new_tokens": max_tokens,
                "min_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
        )

        # ----- Generate response -----
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else "I was unable to generate a response. Please try again."

    except Exception as e:
        # Surface a clean error instead of crashing the app
        return f"⚠️ IBM watsonx.ai error: {str(e)}"


# ===========================================================================
#  AGENT 1 – Nutrition Knowledge Agent
#  Answers general nutrition questions using IBM Granite.
# ===========================================================================
def nutrition_knowledge_agent(question: str) -> str:
    """
    Agent 1: Answers nutrition-related knowledge questions.
    Builds a concise educational prompt and sends it to IBM watsonx.ai.
    """
    prompt = f"""You are NutriWise, an expert nutrition coach powered by IBM watsonx.ai.
Answer the following nutrition question in a clear, educational, and helpful way.
Structure your answer with:
1. A brief overview (2-3 sentences)
2. Key nutritional benefits or facts (bullet points)
3. Practical tips or serving suggestions
4. Any important cautions or notes

Question: {question}

Answer:"""
    # ---> IBM watsonx.ai Granite call
    return generate_response(prompt, max_tokens=800)


# ===========================================================================
#  AGENT 2 – Diet Planner Agent
#  Generates a personalised meal plan using IBM Granite.
# ===========================================================================
def diet_planner_agent(age: str, gender: str, height: str, weight: str,
                       dietary_pref: str, activity_level: str, goal: str) -> str:
    """
    Agent 2: Creates a personalised 1-day meal plan with macro recommendations.
    Passes user profile to IBM watsonx.ai and returns a structured plan.
    """
    prompt = f"""You are NutriWise, a certified AI dietitian powered by IBM watsonx.ai.
Create a detailed, personalised 1-day meal plan for the following individual.

--- User Profile ---
Age            : {age} years
Gender         : {gender}
Height         : {height} cm
Weight         : {weight} kg
Dietary Pref   : {dietary_pref}
Activity Level : {activity_level}
Fitness Goal   : {goal}

--- Your Output Should Include ---
1. Daily Calorie Target (estimated)
2. Macronutrient Recommendations: Protein (g), Carbohydrates (g), Fats (g)
3. Meal Plan:
   🌅 Breakfast (with portion sizes)
   🌞 Mid-Morning Snack
   🍽️ Lunch (with portion sizes)
   🍎 Evening Snack
   🌙 Dinner (with portion sizes)
4. Hydration tip
5. One motivational note aligned with the goal

Be specific, practical, and culturally inclusive. Use Indian foods where appropriate.

Meal Plan:"""
    # ---> IBM watsonx.ai Granite call
    return generate_response(prompt, max_tokens=1024)


# ===========================================================================
#  AGENT 3 – Health Advisory Agent
#  Provides disease-specific dietary guidance using IBM Granite.
# ===========================================================================
def health_advisory_agent(conditions: list) -> str:
    """
    Agent 3: Generates dietary and lifestyle advice for selected health conditions.
    Always appends a medical disclaimer.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Health"

    prompt = f"""You are NutriWise, an AI health advisor powered by IBM watsonx.ai.
Provide comprehensive dietary and lifestyle guidance for someone managing the following condition(s): {conditions_str}.

Structure your response exactly as follows:

✅ FOODS TO INCLUDE
(List 6-8 recommended foods with brief reasons)

❌ FOODS TO AVOID
(List 6-8 foods to avoid or limit with brief reasons)

💚 HEALTHY HABITS
(List 4-5 daily habits that support these conditions)

🌟 LIFESTYLE RECOMMENDATIONS
(3-4 broader lifestyle tips)

Be evidence-based, clear, and compassionate. Use simple language.

Health Advisory:"""
    # ---> IBM watsonx.ai Granite call
    result = generate_response(prompt, max_tokens=900)

    # Always append the medical disclaimer for health advice
    disclaimer = (
        "\n\n---\n"
        "⚕️ **Disclaimer:** This is educational information only. "
        "Always consult a qualified healthcare professional or registered dietitian "
        "before making changes to your diet or lifestyle, especially when managing a medical condition."
    )
    return result + disclaimer


# ===========================================================================
#  AGENT 4 – Meal Analysis Agent
#  Analyses a free-text meal log and provides AI-powered feedback.
# ===========================================================================
def meal_analysis_agent(meal_log: str) -> str:
    """
    Agent 4: Analyses a free-text meal description.
    Evaluates nutritional quality, strengths, gaps, and improvements.
    Powered by IBM watsonx.ai Granite.
    """
    prompt = f"""You are NutriWise, an AI meal analysis expert powered by IBM watsonx.ai.
Analyse the following meal log and provide detailed nutritional feedback.

--- Meal Log ---
{meal_log}

--- Your Analysis Should Include ---
1. 📊 NUTRITIONAL OVERVIEW
   - Estimated calorie range
   - Macronutrient balance assessment (protein/carbs/fats)
   - Micronutrient highlights

2. 💪 NUTRITIONAL STRENGTHS
   (What this meal does well – list 3-4 points)

3. ⚠️ NUTRITIONAL GAPS / DEFICIENCIES
   (What is missing or insufficient – list 3-4 points)

4. 🔄 HEALTHIER ALTERNATIVES
   (Suggest swaps for 2-3 specific items in the meal)

5. 🎯 IMPROVEMENT RECOMMENDATIONS
   (3-4 actionable tips to make this day's eating healthier)

6. ⭐ OVERALL SCORE: X/10 with a one-sentence verdict

Be constructive, encouraging, and practical.

Meal Analysis:"""
    # ---> IBM watsonx.ai Granite call
    return generate_response(prompt, max_tokens=1000)


# ===========================================================================
#  AGENT ORCHESTRATOR
#  Routes incoming requests to the appropriate specialised agent.
# ===========================================================================
def orchestrator(agent_name: str, payload: dict) -> str:
    """
    Central orchestrator that dispatches tasks to the correct agent.

    Agents
    ------
    nutrition_knowledge – Agent 1
    diet_planner        – Agent 2
    health_advisory     – Agent 3
    meal_analysis       – Agent 4
    """
    if agent_name == "nutrition_knowledge":
        return nutrition_knowledge_agent(payload.get("question", ""))

    elif agent_name == "diet_planner":
        return diet_planner_agent(
            age=payload.get("age", "25"),
            gender=payload.get("gender", "Male"),
            height=payload.get("height", "170"),
            weight=payload.get("weight", "70"),
            dietary_pref=payload.get("dietary_pref", "Vegetarian"),
            activity_level=payload.get("activity_level", "Moderate"),
            goal=payload.get("goal", "General Wellness"),
        )

    elif agent_name == "health_advisory":
        return health_advisory_agent(payload.get("conditions", []))

    elif agent_name == "meal_analysis":
        return meal_analysis_agent(payload.get("meal_log", ""))

    else:
        return "⚠️ Unknown agent requested."


# ===========================================================================
#  HTML TEMPLATES (all inlined via render_template_string)
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared base layout – sidebar + Bootstrap 5
# ---------------------------------------------------------------------------
BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NutriWise AI – {{ page_title }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
  <style>
    :root {
      --brand-green : #2e7d32;
      --brand-light : #e8f5e9;
      --brand-teal  : #00695c;
      --sidebar-w   : 260px;
      --accent      : #43a047;
    }
    body { font-family: "Segoe UI", system-ui, sans-serif; background:#f4f6f4; margin:0; }

    /* ---- Sidebar ---- */
    #sidebar {
      width: var(--sidebar-w); height:100vh; position:fixed; top:0; left:0;
      background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 60%, #00695c 100%);
      overflow-y:auto; z-index:100; display:flex; flex-direction:column;
    }
    #sidebar .brand {
      padding:24px 20px 16px; border-bottom:1px solid rgba(255,255,255,.15);
    }
    #sidebar .brand h5 { color:#fff; font-weight:700; margin:0; font-size:1.05rem; letter-spacing:.3px; }
    #sidebar .brand small { color:rgba(255,255,255,.65); font-size:.75rem; }
    #sidebar .nav-link {
      color:rgba(255,255,255,.8); padding:10px 20px; border-radius:0;
      display:flex; align-items:center; gap:10px; font-size:.88rem; transition:all .2s;
    }
    #sidebar .nav-link:hover, #sidebar .nav-link.active {
      background:rgba(255,255,255,.15); color:#fff;
    }
    #sidebar .nav-link i { font-size:1rem; width:20px; text-align:center; }
    #sidebar .section-label {
      color:rgba(255,255,255,.45); font-size:.68rem; text-transform:uppercase;
      letter-spacing:1px; padding:16px 20px 4px; font-weight:600;
    }
    #sidebar .ibm-badge {
      margin:auto 16px 20px; padding:10px 14px;
      background:rgba(255,255,255,.1); border-radius:8px;
      color:rgba(255,255,255,.8); font-size:.73rem; text-align:center; line-height:1.5;
    }

    /* ---- Main content ---- */
    #main { margin-left:var(--sidebar-w); min-height:100vh; }
    .top-bar {
      background:#fff; border-bottom:1px solid #e0e0e0; padding:14px 28px;
      display:flex; align-items:center; justify-content:space-between;
      position:sticky; top:0; z-index:50;
    }
    .top-bar h6 { margin:0; font-weight:600; color:#2e7d32; }
    .content-area { padding:28px; }

    /* ---- Cards ---- */
    .agent-card {
      background:#fff; border-radius:14px; border:1px solid #e0e0e0;
      padding:24px; margin-bottom:24px; transition:box-shadow .2s;
    }
    .agent-card:hover { box-shadow:0 4px 20px rgba(46,125,50,.12); }
    .agent-card .card-icon {
      width:48px; height:48px; border-radius:12px;
      display:flex; align-items:center; justify-content:center;
      font-size:1.4rem; margin-bottom:14px;
    }
    .icon-green  { background:#e8f5e9; color:#2e7d32; }
    .icon-blue   { background:#e3f2fd; color:#1565c0; }
    .icon-orange { background:#fff3e0; color:#e65100; }
    .icon-purple { background:#f3e5f5; color:#6a1b9a; }

    /* ---- Chat / Response area ---- */
    .response-box {
      background:#f9fbe7; border:1px solid #c5e1a5; border-radius:10px;
      padding:20px; margin-top:20px; white-space:pre-wrap;
      font-size:.88rem; line-height:1.7; min-height:80px; color:#1b2a1b;
    }
    .response-box.loading { background:#f0f4f0; color:#78909c; font-style:italic; }

    /* ---- Form controls ---- */
    .form-label { font-size:.83rem; font-weight:600; color:#37474f; }
    .btn-brand {
      background:linear-gradient(135deg, #2e7d32, #43a047);
      color:#fff; border:none; padding:10px 28px; border-radius:8px;
      font-weight:600; letter-spacing:.3px; transition:opacity .2s;
    }
    .btn-brand:hover { opacity:.88; color:#fff; }
    .btn-brand:disabled { opacity:.55; }

    /* ---- Home hero ---- */
    .hero-banner {
      background:linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #00695c 100%);
      color:#fff; padding:44px 36px; border-radius:16px; margin-bottom:28px;
    }
    .hero-banner h2 { font-weight:800; font-size:1.8rem; }
    .hero-banner p  { opacity:.88; margin:0; font-size:.95rem; }

    .stat-pill {
      background:rgba(255,255,255,.18); border-radius:24px;
      padding:6px 16px; font-size:.8rem; display:inline-block; margin:4px 4px 0 0;
    }

    /* ---- About page ---- */
    .arch-step {
      display:flex; gap:16px; padding:16px 0; border-bottom:1px solid #eee;
    }
    .arch-step:last-child { border-bottom:none; }
    .arch-num {
      width:36px; height:36px; border-radius:50%; background:#2e7d32;
      color:#fff; display:flex; align-items:center; justify-content:center;
      font-weight:700; font-size:.9rem; flex-shrink:0;
    }

    /* ---- Checkboxes ---- */
    .condition-check .form-check {
      background:#f9f9f9; border:1px solid #e0e0e0; border-radius:8px;
      padding:10px 14px 10px 36px; margin-bottom:8px; transition:background .15s;
    }
    .condition-check .form-check:hover { background:#e8f5e9; }

    /* ---- Responsive ---- */
    @media(max-width:768px){
      #sidebar { width:100%; height:auto; position:relative; }
      #main { margin-left:0; }
    }
  </style>
</head>
<body>

<!-- ============ SIDEBAR ============ -->
<div id="sidebar">
  <div class="brand">
    <h5><i class="bi bi-activity me-2"></i>NutriWise AI</h5>
    <small>Personalized Nutrition Coach</small>
  </div>

  <div class="mt-2">
    <div class="section-label">Navigation</div>
    <a href="/" class="nav-link {{ 'active' if active=='home' }}">
      <i class="bi bi-house-door"></i> Home
    </a>

    <div class="section-label">AI Agents</div>
    <a href="/nutrition-chat" class="nav-link {{ 'active' if active=='nutrition' }}">
      <i class="bi bi-chat-dots"></i> Nutrition Chat
    </a>
    <a href="/diet-planner" class="nav-link {{ 'active' if active=='diet' }}">
      <i class="bi bi-calendar2-heart"></i> Diet Planner
    </a>
    <a href="/health-advisor" class="nav-link {{ 'active' if active=='health' }}">
      <i class="bi bi-heart-pulse"></i> Health Advisor
    </a>
    <a href="/meal-analyzer" class="nav-link {{ 'active' if active=='meal' }}">
      <i class="bi bi-clipboard2-data"></i> Meal Analyzer
    </a>

    <div class="section-label">Info</div>
    <a href="/about" class="nav-link {{ 'active' if active=='about' }}">
      <i class="bi bi-info-circle"></i> About
    </a>
  </div>

  <div class="ibm-badge mt-auto">
    <div><strong>Powered by</strong></div>
    <div>IBM watsonx.ai</div>
    <div style="opacity:.6">Granite Models</div>
  </div>
</div>
<!-- ============ END SIDEBAR ============ -->

<!-- ============ MAIN ============ -->
<div id="main">
  <div class="top-bar">
    <h6><i class="bi bi-activity me-2"></i>{{ page_title }}</h6>
    <span style="font-size:.78rem;color:#78909c;">
      <i class="bi bi-cpu me-1"></i>ibm/granite-13b-instruct-v2
    </span>
  </div>
  <div class="content-area">
    {% block content %}{% endblock %}
  </div>
</div>
<!-- ============ END MAIN ============ -->

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Home Page
# ---------------------------------------------------------------------------
HOME_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="hero-banner">
  <h2><i class="bi bi-activity me-2"></i>NutriWise AI</h2>
  <p class="mt-2 mb-3">Your intelligent, multi-agent nutrition coach — powered by IBM watsonx.ai Granite Models.</p>
  <span class="stat-pill"><i class="bi bi-robot me-1"></i>4 Specialized AI Agents</span>
  <span class="stat-pill"><i class="bi bi-cpu me-1"></i>IBM Granite 13B</span>
  <span class="stat-pill"><i class="bi bi-shield-check me-1"></i>Health-Safe AI</span>
  <span class="stat-pill"><i class="bi bi-lightning me-1"></i>Real-time Insights</span>
</div>

<div class="row g-4">
  <div class="col-md-6 col-xl-3">
    <a href="/nutrition-chat" class="text-decoration-none">
      <div class="agent-card h-100">
        <div class="card-icon icon-green"><i class="bi bi-chat-dots"></i></div>
        <h6 class="fw-700">Nutrition Knowledge</h6>
        <p class="text-muted small mb-3">Ask any nutrition question and get AI-powered educational answers from IBM Granite.</p>
        <span class="badge" style="background:#e8f5e9;color:#2e7d32;">Agent 1</span>
      </div>
    </a>
  </div>
  <div class="col-md-6 col-xl-3">
    <a href="/diet-planner" class="text-decoration-none">
      <div class="agent-card h-100">
        <div class="card-icon icon-blue"><i class="bi bi-calendar2-heart"></i></div>
        <h6 class="fw-700">Diet Planner</h6>
        <p class="text-muted small mb-3">Generate a personalised meal plan based on your age, goal, and dietary preference.</p>
        <span class="badge" style="background:#e3f2fd;color:#1565c0;">Agent 2</span>
      </div>
    </a>
  </div>
  <div class="col-md-6 col-xl-3">
    <a href="/health-advisor" class="text-decoration-none">
      <div class="agent-card h-100">
        <div class="card-icon icon-orange"><i class="bi bi-heart-pulse"></i></div>
        <h6 class="fw-700">Health Advisor</h6>
        <p class="text-muted small mb-3">Get disease-specific dietary guidance for Diabetes, Hypertension, PCOS & more.</p>
        <span class="badge" style="background:#fff3e0;color:#e65100;">Agent 3</span>
      </div>
    </a>
  </div>
  <div class="col-md-6 col-xl-3">
    <a href="/meal-analyzer" class="text-decoration-none">
      <div class="agent-card h-100">
        <div class="card-icon icon-purple"><i class="bi bi-clipboard2-data"></i></div>
        <h6 class="fw-700">Meal Analyzer</h6>
        <p class="text-muted small mb-3">Enter your meals in plain text and get AI-powered nutritional analysis & tips.</p>
        <span class="badge" style="background:#f3e5f5;color:#6a1b9a;">Agent 4</span>
      </div>
    </a>
  </div>
</div>

<div class="agent-card mt-2">
  <div class="row align-items-center">
    <div class="col-md-8">
      <h6 class="mb-1"><i class="bi bi-info-circle text-success me-2"></i>How NutriWise AI Works</h6>
      <p class="text-muted small mb-0">
        NutriWise AI uses an <strong>Agent Orchestrator</strong> to route your request to one of four
        specialised IBM Granite-powered agents. Each agent constructs a context-rich prompt, sends it to
        <code>ibm/granite-13b-instruct-v2</code> via the watsonx.ai API, and returns a structured,
        personalised response — all in real time.
      </p>
    </div>
    <div class="col-md-4 text-md-end mt-3 mt-md-0">
      <a href="/about" class="btn btn-outline-success btn-sm">View Architecture →</a>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")

# ---------------------------------------------------------------------------
# Nutrition Chat Page
# ---------------------------------------------------------------------------
NUTRITION_CHAT_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="agent-card">
  <div class="d-flex align-items-center gap-3 mb-3">
    <div class="card-icon icon-green mb-0"><i class="bi bi-chat-dots"></i></div>
    <div>
      <h5 class="mb-0 fw-bold">Nutrition Knowledge Agent</h5>
      <small class="text-muted">Agent 1 · IBM Granite 13B · Ask any nutrition question</small>
    </div>
  </div>

  <div class="mb-3">
    <label class="form-label">Your Nutrition Question</label>
    <textarea id="questionInput" class="form-control" rows="3"
      placeholder="e.g. What are the benefits of oats? Which foods are rich in Vitamin B12? Is paneer healthy for muscle gain?"></textarea>
  </div>

  <div class="d-flex gap-2 flex-wrap mb-2">
    <button class="btn btn-sm btn-outline-success" onclick="fillQ('What are the health benefits of oats?')">🌾 Benefits of oats</button>
    <button class="btn btn-sm btn-outline-success" onclick="fillQ('Which foods are rich in protein for vegetarians?')">🥜 Protein sources</button>
    <button class="btn btn-sm btn-outline-success" onclick="fillQ('What foods contain Vitamin B12?')">💊 Vitamin B12</button>
    <button class="btn btn-sm btn-outline-success" onclick="fillQ('Is paneer healthy for muscle gain?')">🧀 Paneer & muscles</button>
  </div>

  <button class="btn btn-brand mt-2" id="askBtn" onclick="askQuestion()">
    <i class="bi bi-send me-2"></i>Ask NutriWise AI
  </button>

  <div id="responseBox" class="response-box" style="display:none;"></div>
</div>

<div class="agent-card">
  <h6 class="mb-3"><i class="bi bi-lightbulb text-success me-2"></i>Example Questions to Try</h6>
  <div class="row g-2">
    {% for q in example_questions %}
    <div class="col-md-6">
      <div class="p-2 rounded" style="background:#f9fbe7;border:1px solid #dcedc8;font-size:.82rem;cursor:pointer;"
           onclick="fillQ('{{ q }}')">
        <i class="bi bi-arrow-right-circle text-success me-2"></i>{{ q }}
      </div>
    </div>
    {% endfor %}
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
function fillQ(text){ document.getElementById('questionInput').value = text; }

async function askQuestion(){
  const q = document.getElementById('questionInput').value.trim();
  if(!q){ alert('Please enter a question.'); return; }
  const btn = document.getElementById('askBtn');
  const box = document.getElementById('responseBox');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Consulting IBM Granite...';
  box.style.display='block';
  box.className='response-box loading';
  box.innerText='⏳ IBM watsonx.ai Granite is generating your answer...';
  try {
    const res = await fetch('/api/nutrition-knowledge', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    box.className='response-box';
    box.innerText = data.response || 'No response received.';
  } catch(e) {
    box.className='response-box';
    box.innerText = '⚠️ Error communicating with the server: ' + e.message;
  }
  btn.disabled=false;
  btn.innerHTML='<i class="bi bi-send me-2"></i>Ask NutriWise AI';
}
</script>
""")

# ---------------------------------------------------------------------------
# Diet Planner Page
# ---------------------------------------------------------------------------
DIET_PLANNER_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="agent-card">
  <div class="d-flex align-items-center gap-3 mb-4">
    <div class="card-icon icon-blue mb-0"><i class="bi bi-calendar2-heart"></i></div>
    <div>
      <h5 class="mb-0 fw-bold">Diet Planner Agent</h5>
      <small class="text-muted">Agent 2 · IBM Granite 13B · Personalised meal planning</small>
    </div>
  </div>

  <div class="row g-3">
    <div class="col-md-4">
      <label class="form-label">Age (years)</label>
      <input type="number" id="age" class="form-control" placeholder="e.g. 25" min="10" max="100"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Gender</label>
      <select id="gender" class="form-select">
        <option>Male</option><option>Female</option><option>Other</option>
      </select>
    </div>
    <div class="col-md-4">
      <label class="form-label">Height (cm)</label>
      <input type="number" id="height" class="form-control" placeholder="e.g. 170" min="100" max="250"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Weight (kg)</label>
      <input type="number" id="weight" class="form-control" placeholder="e.g. 70" min="30" max="300"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Dietary Preference</label>
      <select id="dietary_pref" class="form-select">
        <option>Vegetarian</option>
        <option>Vegan</option>
        <option>Non-Vegetarian</option>
        <option>Eggetarian</option>
        <option>Jain</option>
      </select>
    </div>
    <div class="col-md-4">
      <label class="form-label">Activity Level</label>
      <select id="activity_level" class="form-select">
        <option>Sedentary (little/no exercise)</option>
        <option>Lightly Active (1-3 days/week)</option>
        <option>Moderate (3-5 days/week)</option>
        <option>Very Active (6-7 days/week)</option>
        <option>Athlete / Intense training</option>
      </select>
    </div>
    <div class="col-12">
      <label class="form-label">Fitness Goal</label>
      <div class="d-flex gap-2 flex-wrap">
        {% for goal in goals %}
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="goal_{{ loop.index }}" value="{{ goal }}"
            {{ 'checked' if loop.first }}>
          <label class="form-check-label small" for="goal_{{ loop.index }}">{{ goal }}</label>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <button class="btn btn-brand mt-4" id="planBtn" onclick="generatePlan()">
    <i class="bi bi-calendar-check me-2"></i>Generate My Meal Plan
  </button>

  <div id="planResponse" class="response-box" style="display:none;"></div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
async function generatePlan(){
  const age = document.getElementById('age').value;
  const gender = document.getElementById('gender').value;
  const height = document.getElementById('height').value;
  const weight = document.getElementById('weight').value;
  const dietary_pref = document.getElementById('dietary_pref').value;
  const activity_level = document.getElementById('activity_level').value;
  const goalEl = document.querySelector('input[name="goal"]:checked');
  const goal = goalEl ? goalEl.value : 'General Wellness';

  if(!age || !height || !weight){ alert('Please fill in Age, Height, and Weight.'); return; }

  const btn = document.getElementById('planBtn');
  const box = document.getElementById('planResponse');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner-border spinner-border-sm me-2"></span>IBM Granite is crafting your plan...';
  box.style.display='block';
  box.className='response-box loading';
  box.innerText='⏳ Generating your personalised meal plan via IBM watsonx.ai...';

  try {
    const res = await fetch('/api/diet-planner', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({age, gender, height, weight, dietary_pref, activity_level, goal})
    });
    const data = await res.json();
    box.className='response-box';
    box.innerText = data.response || 'No response received.';
  } catch(e) {
    box.className='response-box';
    box.innerText = '⚠️ Error: ' + e.message;
  }
  btn.disabled=false;
  btn.innerHTML='<i class="bi bi-calendar-check me-2"></i>Generate My Meal Plan';
}
</script>
""")

# ---------------------------------------------------------------------------
# Health Advisor Page
# ---------------------------------------------------------------------------
HEALTH_ADVISOR_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="agent-card">
  <div class="d-flex align-items-center gap-3 mb-4">
    <div class="card-icon icon-orange mb-0"><i class="bi bi-heart-pulse"></i></div>
    <div>
      <h5 class="mb-0 fw-bold">Health Advisory Agent</h5>
      <small class="text-muted">Agent 3 · IBM Granite 13B · Disease-specific dietary guidance</small>
    </div>
  </div>

  <p class="text-muted small mb-3">Select one or more health conditions to receive personalised dietary and lifestyle recommendations.</p>

  <div class="condition-check row g-2">
    {% for condition in conditions %}
    <div class="col-md-6">
      <div class="form-check">
        <input class="form-check-input" type="checkbox" id="cond_{{ loop.index }}" value="{{ condition }}">
        <label class="form-check-label small fw-semibold" for="cond_{{ loop.index }}">
          {{ condition }}
        </label>
      </div>
    </div>
    {% endfor %}
  </div>

  <button class="btn btn-brand mt-4" id="advBtn" onclick="getAdvice()">
    <i class="bi bi-shield-heart me-2"></i>Get Health Advice
  </button>

  <div id="advResponse" class="response-box" style="display:none;"></div>

  <div class="mt-3 p-3 rounded" style="background:#fff8e1;border:1px solid #ffe082;font-size:.8rem;color:#5d4037;">
    <i class="bi bi-exclamation-triangle me-2"></i>
    <strong>Disclaimer:</strong> Information provided is for educational purposes only.
    Always consult a qualified healthcare professional before modifying your diet or lifestyle.
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
async function getAdvice(){
  const checked = [...document.querySelectorAll('.form-check-input:checked')].map(c=>c.value);
  if(checked.length===0){ alert('Please select at least one health condition.'); return; }

  const btn = document.getElementById('advBtn');
  const box = document.getElementById('advResponse');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner-border spinner-border-sm me-2"></span>IBM Granite is preparing advice...';
  box.style.display='block';
  box.className='response-box loading';
  box.innerText='⏳ Generating health advisory via IBM watsonx.ai...';

  try {
    const res = await fetch('/api/health-advisory', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({conditions: checked})
    });
    const data = await res.json();
    box.className='response-box';
    box.innerText = data.response || 'No response received.';
  } catch(e) {
    box.className='response-box';
    box.innerText = '⚠️ Error: ' + e.message;
  }
  btn.disabled=false;
  btn.innerHTML='<i class="bi bi-shield-heart me-2"></i>Get Health Advice';
}
</script>
""")

# ---------------------------------------------------------------------------
# Meal Analyzer Page
# ---------------------------------------------------------------------------
MEAL_ANALYZER_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="agent-card">
  <div class="d-flex align-items-center gap-3 mb-4">
    <div class="card-icon icon-purple mb-0"><i class="bi bi-clipboard2-data"></i></div>
    <div>
      <h5 class="mb-0 fw-bold">Meal Analysis Agent</h5>
      <small class="text-muted">Agent 4 · IBM Granite 13B · Free-text meal nutritional analysis</small>
    </div>
  </div>

  <label class="form-label">Enter Your Meals for the Day</label>
  <textarea id="mealLog" class="form-control" rows="8"
    placeholder="Breakfast:&#10;  2 Rotis&#10;  Dal (1 bowl)&#10;  Curd&#10;&#10;Lunch:&#10;  Rice (1 cup)&#10;  Paneer Curry&#10;  Salad&#10;&#10;Snacks:&#10;  Banana, Tea with milk&#10;&#10;Dinner:&#10;  Khichdi (1 bowl)&#10;  Vegetable soup"></textarea>

  <div class="d-flex gap-2 flex-wrap mt-2 mb-2">
    <button class="btn btn-sm btn-outline-secondary" onclick="loadSample('veg')">🥗 Load Veg Sample</button>
    <button class="btn btn-sm btn-outline-secondary" onclick="loadSample('nonveg')">🍗 Load Non-Veg Sample</button>
    <button class="btn btn-sm btn-outline-secondary" onclick="document.getElementById('mealLog').value=''">🗑 Clear</button>
  </div>

  <button class="btn btn-brand mt-2" id="analyzeBtn" onclick="analyzeMeal()">
    <i class="bi bi-bar-chart me-2"></i>Analyze My Meals
  </button>

  <div id="mealResponse" class="response-box" style="display:none;"></div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
const samples = {
  veg: `Breakfast:\\n  2 Rotis\\n  Dal (1 katori)\\n  Curd (100g)\\n\\nLunch:\\n  Rice (1 cup)\\n  Rajma curry\\n  Mixed salad\\n\\nSnacks:\\n  1 Apple, Green tea\\n\\nDinner:\\n  Khichdi (1 bowl)\\n  Stir-fried vegetables`,
  nonveg: `Breakfast:\\n  2 Eggs (scrambled)\\n  2 Brown bread slices\\n  Milk (1 glass)\\n\\nLunch:\\n  Rice (1 cup)\\n  Chicken curry (1 serving)\\n  Salad\\n\\nSnacks:\\n  Boiled chickpeas, Black coffee\\n\\nDinner:\\n  Roti (2)\\n  Grilled fish\\n  Dal`
};
function loadSample(type){ document.getElementById('mealLog').value = samples[type]; }

async function analyzeMeal(){
  const log = document.getElementById('mealLog').value.trim();
  if(!log){ alert('Please enter your meals first.'); return; }
  const btn = document.getElementById('analyzeBtn');
  const box = document.getElementById('mealResponse');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner-border spinner-border-sm me-2"></span>IBM Granite is analyzing...';
  box.style.display='block';
  box.className='response-box loading';
  box.innerText='⏳ Performing nutritional analysis via IBM watsonx.ai...';
  try {
    const res = await fetch('/api/meal-analysis', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({meal_log: log})
    });
    const data = await res.json();
    box.className='response-box';
    box.innerText = data.response || 'No response received.';
  } catch(e) {
    box.className='response-box';
    box.innerText = '⚠️ Error: ' + e.message;
  }
  btn.disabled=false;
  btn.innerHTML='<i class="bi bi-bar-chart me-2"></i>Analyze My Meals';
}
</script>
""")

# ---------------------------------------------------------------------------
# About Page
# ---------------------------------------------------------------------------
ABOUT_PAGE = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="agent-card">
  <h4 class="fw-bold mb-1"><i class="bi bi-diagram-3 text-success me-2"></i>About NutriWise AI</h4>
  <p class="text-muted small">Multi-Agent Architecture · IBM watsonx.ai Integration · Built for IBM Hackathons</p>

  <div class="mt-4">
    <h6 class="fw-bold">🏗 Multi-Agent Architecture</h6>
    <p class="text-muted small mb-3">
      NutriWise AI is built on a <strong>4-agent orchestration pattern</strong>.
      A central orchestrator receives the user request, identifies the intent,
      and routes it to the appropriate specialised IBM Granite-powered agent.
    </p>

    <div class="arch-step">
      <div class="arch-num">1</div>
      <div>
        <strong>Nutrition Knowledge Agent</strong>
        <div class="text-muted small mt-1">Answers general nutrition questions. Constructs an educational prompt and queries
        <code>ibm/granite-13b-instruct-v2</code> via the watsonx.ai Python SDK.</div>
      </div>
    </div>
    <div class="arch-step">
      <div class="arch-num">2</div>
      <div>
        <strong>Diet Planner Agent</strong>
        <div class="text-muted small mt-1">Accepts user profile (age, weight, goal, dietary preference) and generates
        a structured day-long meal plan with macro targets using IBM Granite.</div>
      </div>
    </div>
    <div class="arch-step">
      <div class="arch-num">3</div>
      <div>
        <strong>Health Advisory Agent</strong>
        <div class="text-muted small mt-1">Generates condition-specific dietary and lifestyle guidance for Diabetes,
        Hypertension, PCOS, etc. Always includes a medical disclaimer.</div>
      </div>
    </div>
    <div class="arch-step">
      <div class="arch-num">4</div>
      <div>
        <strong>Meal Analysis Agent</strong>
        <div class="text-muted small mt-1">Accepts free-text meal logs and performs nutritional quality assessment —
        scoring, gaps, strengths, swaps, and improvement tips — all via IBM Granite.</div>
      </div>
    </div>
  </div>
</div>

<div class="row g-4 mt-1">
  <div class="col-md-6">
    <div class="agent-card h-100">
      <h6 class="fw-bold"><i class="bi bi-cpu text-success me-2"></i>IBM watsonx.ai Integration</h6>
      <ul class="text-muted small ps-3 mt-2">
        <li>SDK: <code>ibm-watsonx-ai</code></li>
        <li>Model: <code>ibm/granite-13b-instruct-v2</code></li>
        <li>Auth: API Key via environment variable</li>
        <li>All agents share a single <code>generate_response()</code> function</li>
        <li>Credentials: <code>WATSONX_API_KEY</code>, <code>WATSONX_PROJECT_ID</code>, <code>WATSONX_URL</code></li>
      </ul>
    </div>
  </div>
  <div class="col-md-6">
    <div class="agent-card h-100">
      <h6 class="fw-bold"><i class="bi bi-stack text-success me-2"></i>Tech Stack</h6>
      <ul class="text-muted small ps-3 mt-2">
        <li>Backend: <strong>Python 3.10+ · Flask</strong></li>
        <li>AI Engine: <strong>IBM watsonx.ai (Granite)</strong></li>
        <li>Frontend: <strong>Bootstrap 5 · Vanilla JS</strong></li>
        <li>UI Pattern: Sidebar + Card layout</li>
        <li>Deployment: Single <code>app.py</code> file</li>
      </ul>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")


# ===========================================================================
#  FLASK ROUTES
# ===========================================================================

@app.route("/")
def home():
    """Home page – overview of all four agents."""
    return render_template_string(HOME_PAGE, active="home", page_title="Home")


@app.route("/nutrition-chat")
def nutrition_chat():
    """Agent 1 – Nutrition Knowledge Chat page."""
    example_questions = [
        "What are the health benefits of turmeric?",
        "How much protein does a 70kg person need daily?",
        "What are the best foods for improving gut health?",
        "Which foods help reduce inflammation naturally?",
        "What is the difference between simple and complex carbohydrates?",
        "Which Indian foods are high in iron?",
        "How does fibre help with digestion?",
        "What are omega-3 fatty acids and where are they found?",
    ]
    return render_template_string(
        NUTRITION_CHAT_PAGE,
        active="nutrition",
        page_title="Nutrition Chat",
        example_questions=example_questions,
    )


@app.route("/diet-planner")
def diet_planner():
    """Agent 2 – Diet Planner page."""
    goals = ["Weight Loss", "Weight Gain", "Muscle Gain", "General Wellness", "Improve Endurance", "Better Sleep & Recovery"]
    return render_template_string(
        DIET_PLANNER_PAGE,
        active="diet",
        page_title="Diet Planner",
        goals=goals,
    )


@app.route("/health-advisor")
def health_advisor():
    """Agent 3 – Health Advisory page."""
    conditions = [
        "Type 2 Diabetes",
        "Hypertension (High BP)",
        "Obesity",
        "Heart Disease",
        "PCOS / PCOD",
        "High Cholesterol",
        "Thyroid (Hypothyroidism)",
        "Anaemia",
    ]
    return render_template_string(
        HEALTH_ADVISOR_PAGE,
        active="health",
        page_title="Health Advisor",
        conditions=conditions,
    )


@app.route("/meal-analyzer")
def meal_analyzer():
    """Agent 4 – Meal Analyzer page."""
    return render_template_string(
        MEAL_ANALYZER_PAGE,
        active="meal",
        page_title="Meal Analyzer",
    )


@app.route("/about")
def about():
    """About page – architecture and IBM watsonx.ai integration details."""
    return render_template_string(ABOUT_PAGE, active="about", page_title="About NutriWise AI")


# ---------------------------------------------------------------------------
# JSON API endpoints – called by frontend JavaScript via fetch()
# Each endpoint passes the payload to the Agent Orchestrator.
# ---------------------------------------------------------------------------

@app.route("/api/nutrition-knowledge", methods=["POST"])
def api_nutrition_knowledge():
    """REST endpoint for Agent 1 – Nutrition Knowledge Agent."""
    data = request.get_json(force=True)
    # --> Orchestrator routes to nutrition_knowledge_agent()
    response = orchestrator("nutrition_knowledge", {"question": data.get("question", "")})
    return jsonify({"response": response})


@app.route("/api/diet-planner", methods=["POST"])
def api_diet_planner():
    """REST endpoint for Agent 2 – Diet Planner Agent."""
    data = request.get_json(force=True)
    # --> Orchestrator routes to diet_planner_agent()
    response = orchestrator("diet_planner", data)
    return jsonify({"response": response})


@app.route("/api/health-advisory", methods=["POST"])
def api_health_advisory():
    """REST endpoint for Agent 3 – Health Advisory Agent."""
    data = request.get_json(force=True)
    # --> Orchestrator routes to health_advisory_agent()
    response = orchestrator("health_advisory", {"conditions": data.get("conditions", [])})
    return jsonify({"response": response})


@app.route("/api/meal-analysis", methods=["POST"])
def api_meal_analysis():
    """REST endpoint for Agent 4 – Meal Analysis Agent."""
    data = request.get_json(force=True)
    # --> Orchestrator routes to meal_analysis_agent()
    response = orchestrator("meal_analysis", {"meal_log": data.get("meal_log", "")})
    return jsonify({"response": response})


# ===========================================================================
#  Entry Point
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI – Personalized Nutrition Coach")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print("\n  Make sure these environment variables are set:")
    print("    WATSONX_API_KEY    – your IBM watsonx.ai API key")
    print("    WATSONX_PROJECT_ID – your watsonx.ai project ID")
    print("    WATSONX_URL        – (optional) defaults to us-south\n")
    print("  Starting Flask on http://localhost:5000 ...\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
