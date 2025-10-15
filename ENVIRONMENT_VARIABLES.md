# 🔧 LUKi Core Agent - Environment Variables Guide

## Schema Mode Configuration

### LUKI_SCHEMA_MODE

**Purpose:** Controls which response schema the LLM uses

**Values:**
- `minimal` (default) - Uses LUKiMinimalResponse schema
- `full` - Uses LUKiResponse with thought field

**Where to set:**

#### Railway Dashboard:
1. Go to Railway dashboard
2. Select `luki-core-agent` service
3. Click **Variables** tab
4. Click **+ New Variable**
5. Add:
   - **Key:** `LUKI_SCHEMA_MODE`
   - **Value:** `full` (or `minimal`)
6. Click **Deploy**

#### Local Development (.env file):
```bash
# In luki-core-agent/.env
LUKI_SCHEMA_MODE=full
```

### Comparison: Minimal vs Full

| Feature | Minimal Mode | Full Mode |
|---------|-------------|-----------|
| **Schema** | LUKiMinimalResponse | LUKiResponse |
| **Thought field** | ❌ No | ✅ Yes |
| **Max tokens** | 2048/3072/4096 | 3072/6144/8192 |
| **Field description** | Enhanced (after fix) | Rich personality guidance |
| **Speed** | Faster | Slightly slower |
| **Cost** | Lower | ~20% higher |
| **Quality** | High (after fix) | Highest |
| **Personality** | ✅ Full (after fix) | ✅ Full |

### Recommendation

**After today's fixes:**
- **Minimal mode** now has full personality restoration ✅
- Increased tokens prevent truncation ✅
- **Use minimal mode** - good balance of speed/cost/quality

**When to use full mode:**
- You want internal thought logs for debugging
- You want maximum quality
- Cost is not a concern

---

## Other Environment Variables

### LUKI_STRUCTURED_TIMEOUT_LONG
**Default:** `35`
**Purpose:** Timeout in seconds for longer API calls
**Recommended:** `35-60`

### LUKI_AUTOCONTINUE
**Default:** `true`
**Purpose:** Auto-continue truncated responses
**Values:** `true` / `false`

### TOGETHER_API_KEY
**Required:** YES
**Purpose:** Together AI API authentication

### LUKI_PROMPTS_ARCHIVE_B64
**Optional:** Base64-encoded prompts archive
**Purpose:** Bootstrap prompts directory on Railway

---

## Quick Reference

### Default Settings (Good for production):
```bash
LUKI_SCHEMA_MODE=minimal
LUKI_STRUCTURED_TIMEOUT_LONG=35
LUKI_AUTOCONTINUE=true
```

### High Quality Settings (Higher cost):
```bash
LUKI_SCHEMA_MODE=full
LUKI_STRUCTURED_TIMEOUT_LONG=60
LUKI_AUTOCONTINUE=true
```

### Fast/Budget Settings:
```bash
LUKI_SCHEMA_MODE=minimal
LUKI_STRUCTURED_TIMEOUT_LONG=20
LUKI_AUTOCONTINUE=false
```

---

## After Today's Fixes

**You don't need to change anything!**

The default `minimal` mode now has:
- ✅ Full personality guidance
- ✅ Expressions (*chuckles*, *grins*, *nods*)
- ✅ Higher tokens (no truncation)
- ✅ Fast and cost-effective

**Only change to `full` mode if:**
- You want thought logs for debugging
- You want absolute maximum quality
- Cost is not a concern (20% more expensive)

---

## How Changes Work

### 1. Set in Railway
Railway Variables → Automatically injected into container → Available via `os.getenv()`

### 2. Set in .env (local)
.env file → Loaded by python-dotenv → Available via `os.getenv()`

### 3. Railway vs .env Priority
Railway variables **override** .env file values

---

## Verification

**Check current mode in logs:**
```
# Look for this in Railway logs:
🔍 API call parameters: max_tokens=2048, model=openai/gpt-oss-20b

# Memory detection should show:
🧠 Memory detection mode: using MemoryDetectionResponse schema
```

**If you see validation errors:**
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for LUKiMinimalResponse
```

→ Deploy the latest code (fixes this issue!)

---

## Summary

**Question:** "Where do I set that variable?"

**Answer:**
1. **Railway (production):** Railway Dashboard → Variables → Add `LUKI_SCHEMA_MODE=full`
2. **Local (development):** `.env` file → `LUKI_SCHEMA_MODE=full`

**But:** After today's fixes, you don't need to change it! Minimal mode is now excellent.
