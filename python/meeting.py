import re
import difflib
from datetime import datetime
from dateutil import parser
from dateutil.parser import ParserError
import requests
import warnings
warnings.filterwarnings("ignore", message=".*parsed with no year.*")

# === 1. INTENT RECOGNITION ===
MEETING_REGEX = re.compile(
    r"\b(schedule|book|set|arrange|plan|organize|fix|create)\b.*\b(meeting|call|appointment|discussion|review|sync|huddle)\b"
    r"|\b(meeting|call|appointment)\b.*\b(schedule|book|set|arrange|plan|organize|fix)\b",
    re.IGNORECASE
)

def is_meeting_request(text: str):
    return bool(MEETING_REGEX.search(text))


# === 2. MONTH CORRECTION ===
MONTH_MISSPELLINGS = {
    'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april',
    'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'sept': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december',
    'januray': 'january', 'feburary': 'february', 'novemebr': 'november',
    'decemeber': 'december', 'septemeber': 'september'
}

def correct_month(text):
    words = text.lower().split()
    corrected = []
    for word in words:
        if word in MONTH_MISSPELLINGS:
            corrected.append(MONTH_MISSPELLINGS[word])
        elif len(word) >= 3:
            matches = difflib.get_close_matches(word, MONTH_MISSPELLINGS.keys(), n=1, cutoff=0.6)
            if matches and matches[0] in MONTH_MISSPELLINGS:
                corrected.append(MONTH_MISSPELLINGS[matches[0]])
            else:
                # Keep original if no match (dateutil may still handle it)
                corrected.append(word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

# === 3. ROBUST DATETIME EXTRACTION ===
def extract_datetime(text):
    original = text.strip()
    cleaned = correct_month(original)
    
    # Normalize "4pm" → "4 PM", etc.
    cleaned = re.sub(r'(\d{1,2})(?:[:\.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)',
                     lambda m: f"{m.group(1)}:{m.group(2) or '00'} {m.group(3).replace('.', '').upper()}",
                     cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    try:
        dt = parser.parse(cleaned, fuzzy=True)
        now = datetime.now()
        if dt.date() < now.date() and dt.year == now.year:
            dt = dt.replace(year=now.year + 1)
        return dt
    except (ParserError, ValueError, OverflowError):
        pass

    # Fallback regex patterns
    patterns = [
        r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)',
        r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?',
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})'
    ]
    for pattern in patterns:
        match = re.search(pattern, original, re.IGNORECASE)
        if match:
            try:
                if pattern.endswith('(\d{2,4})'):  # numeric date
                    d, m, y = match.groups()
                    y = '20' + y if len(y) == 2 else y
                    dt_str = f"{d} {m} {y}"
                else:
                    if '(\d{1,2}).*([a-z]+)' in pattern or pattern == patterns[0]:
                        day, month = match.group(1), match.group(2)
                    else:
                        month, day = match.group(1), match.group(2)
                    dt_str = f"{day} {month}"
                    dt_str = correct_month(dt_str)
                dt = parser.parse(dt_str, fuzzy=True)
                now = datetime.now()
                if dt.date() < now.date() and dt.year == now.year:
                    dt = dt.replace(year=now.year + 1)
                return dt
            except:
                continue
    return None 

def generate_meeting_description(chat_history, max_words=30):
    if not chat_history:
        return "General discussion regarding the property."

    # ❌ Remove user lines
    cleaned = [line for line in chat_history if not line.lower().startswith("user:")]

    if not cleaned:
        return "General discussion regarding the property."

    text = " ".join(cleaned[-5:])
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    summary = " ".join(words[:max_words])

    if len(summary) < 10:
        summary = "Discussion regarding project details and user requirements."

    return summary

FINAL = False

# === 4. MAIN BOT ===
class MeetingSchedulerBot:
    def __init__(self):
        self.user_id = None
        self.reset_state()
    
    def reset_state(self):
        self.meeting = {'datetime': None, 'duration_minutes': 30, 'purpose': ''}
        self.awaiting = None
    
    def generate_auto_description(self, chat_history):
        return generate_meeting_description(chat_history, max_words=30)
    
    def respond(self, user_input, chat_history=None):
        user_input = user_input.strip()

        # 🔥 VERY IMPORTANT — store chat history
        self.chat_history = chat_history or []
        
        if self.awaiting == 'datetime':
            dt = extract_datetime(user_input)
            if dt:
                self.meeting['datetime'] = dt
                self.awaiting = 'purpose'
                return f"Got it! to Schedule meeting or confirm on **{dt.strftime('%A, %B %d, %Y at %I:%M %p')}** please Type Yes "
            else:
                return "I couldn't understand the date/time. Try: '28 November', 'Nov 28 at 3 PM', etc."

        elif self.awaiting == 'purpose':

            # Auto-generate description from recent chat
            purpose = self.generate_auto_description(self.chat_history)

            duration = 30  # default duration

            # Detect duration from user text (if user mentions)
            num_match = re.search(r'(\d+)\s*(?:minute|minutes|hour|hours|min|hr)', user_input, re.IGNORECASE)
            if num_match:
                num = int(num_match.group(1))
                if any(w in user_input.lower() for w in ['hour', 'hr']):
                    duration = num * 60
                else:
                    duration = num

            self.meeting['duration_minutes'] = duration
            self.meeting['purpose'] = purpose
            self.awaiting = None

            dt = self.meeting['datetime']

            if self.user_id is None:
                return "❌ Could not get the user ID. Please login again or pass a valid user_id."

            formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")

            payload = {
                "user_id": self.user_id,
                "meeting_date": formatted_date,
                "description": purpose
            }

            try:
                response = requests.post(
                    "http://3.6.203.180:7601/meetings/scheduleMeeting",
                    json=payload
                )
                api_status = f"📡 API Status: {response.status_code}"
            except Exception as e:
                api_status = f"❌ API Error: {str(e)}"

            summary = (
                f"✅ **Meeting Scheduled!**\n"
                f"📅 {dt.strftime('%A, %B %d, %Y')}\n"
                f"⏰ {dt.strftime('%I:%M %p')}\n"
                f"📝 {purpose}\n\n"
            )

            self.reset_state()
            return summary

        if is_meeting_request(user_input):
            self.awaiting = 'datetime'
            return "Sure! Tell me the **date and time** for your meeting (e.g., '28 November at 3 PM')."
        else:
            return "Hello! Say something like 'schedule a meeting' or 'book a call'."

