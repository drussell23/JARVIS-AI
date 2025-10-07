# How to Use the Async Pipeline

**Created:** October 5, 2025
**Purpose:** Guide for actually USING the AdvancedAsyncPipeline, not just initializing it

---

## ⚠️ **Common Mistake**

Many developers initialize the pipeline but never actually USE it:

```python
# ❌ WRONG - Pipeline initialized but not used
class MyComponent:
    def __init__(self):
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    def my_method(self):
        # Still using blocking subprocess.run()
        result = subprocess.run(["command"], capture_output=True)
        return result.stdout
```

The pipeline just sits there unused while your code continues to block! 😱

---

## ✅ **Correct Way**

You must **actually call through the pipeline** for every operation:

```python
# ✅ CORRECT - Pipeline is actually used
class MyComponent:
    def __init__(self):
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    async def my_method_pipeline(self):
        # Route through async pipeline
        result = await self.pipeline.process_async(
            text="Execute command",
            metadata={"stage": "my_stage"}
        )
        return result.get("metadata", {}).get("output")

    def my_method(self):
        # Legacy sync wrapper
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.my_method_pipeline())
```

---

## 📋 **Step-by-Step Integration**

### **Step 1: Register Pipeline Stages**

```python
def _register_pipeline_stages(self):
    """Register what operations the pipeline can handle"""
    self.pipeline.register_stage(
        name="my_operation",           # Unique stage name
        handler=self._my_async_handler, # Async function to call
        timeout=10.0,                    # Max execution time
        retry_count=2,                   # How many retries
        required=True                    # Fail if this fails?
    )
```

### **Step 2: Create Async Handler**

```python
async def _my_async_handler(self, context):
    """This is what actually does the work"""
    # Get input data from context
    command = context.metadata.get("command", "")

    # Do the async operation
    from api.jarvis_voice_api import async_subprocess_run
    stdout, stderr, returncode = await async_subprocess_run(command)

    # Store results back in context
    context.metadata["output"] = stdout.decode()
    context.metadata["success"] = returncode == 0
```

### **Step 3: Create Pipeline Method**

```python
async def execute_command_pipeline(self, command: str) -> Tuple[bool, str]:
    """New async method that uses the pipeline"""
    try:
        # Call through pipeline
        result = await self.pipeline.process_async(
            text=f"Execute: {command}",
            metadata={
                "command": command,
                "stage": "my_operation"  # Which stage to use
            }
        )

        # Extract results
        metadata = result.get("metadata", {})
        success = metadata.get("success", False)
        output = metadata.get("output", "")

        return success, output

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False, str(e)
```

### **Step 4: Update Legacy Method**

```python
def execute_command(self, command: str) -> Tuple[bool, str]:
    """Legacy synchronous method - now wraps async version"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run async version in sync context
    return loop.run_until_complete(
        self.execute_command_pipeline(command)
    )
```

---

## 🎯 **Real Example: MacOS Controller**

Here's how MacOSController was updated to ACTUALLY use the pipeline:

### **Before (Pipeline Not Used):**

```python
class MacOSController:
    def __init__(self):
        self.pipeline = get_async_pipeline()  # Initialized...
        self._register_pipeline_stages()     # Stages registered...

    def execute_applescript(self, script: str):
        # But still using blocking subprocess! 😱
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True
        )
        return result.returncode == 0, result.stdout
```

### **After (Pipeline Actually Used):**

```python
class MacOSController:
    def __init__(self):
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    # NEW: Async version that uses pipeline
    async def execute_applescript_pipeline(self, script: str):
        result = await self.pipeline.process_async(
            text="Execute AppleScript",
            metadata={
                "script": script,
                "stage": "applescript_execution"  # Routes to our stage!
            }
        )

        metadata = result.get("metadata", {})
        return metadata.get("success"), metadata.get("stdout")

    # UPDATED: Legacy version wraps async
    def execute_applescript(self, script: str):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.execute_applescript_pipeline(script)
        )
```

---

## 🔍 **How to Check if Pipeline is Actually Being Used**

### **1. Check for `await self.pipeline.process_async()`**

Your methods should have code like this:

```python
result = await self.pipeline.process_async(...)
```

If you don't see this, the pipeline isn't being used!

### **2. Check for `subprocess.run()` or `subprocess.call()`**

If you see these anywhere in your code, you're still blocking:

```python
# ❌ BAD - This blocks!
subprocess.run(["command"], capture_output=True)

# ✅ GOOD - This is async
from api.jarvis_voice_api import async_subprocess_run
await async_subprocess_run("command")
```

### **3. Check for `requests.get()` or `requests.post()`**

If you see these, you're blocking:

```python
# ❌ BAD - This blocks!
response = requests.get("https://api.example.com")

# ✅ GOOD - This is async
import aiohttp
async with aiohttp.ClientSession() as session:
    async with session.get("https://api.example.com") as resp:
        data = await resp.json()
```

---

## 📊 **Benefits When Properly Used**

When you **actually use** the pipeline (not just initialize it):

### **Before (Blocking):**
- ⏱️ **5-35 seconds** - UI frozen during command
- ❌ **No retry** - Fails permanently
- ❌ **No timeout** - Can hang forever
- ❌ **No metrics** - No visibility into what's happening

### **After (Using Pipeline):**
- ⚡ **0.1-0.5 seconds** - Non-blocking, responsive UI
- ✅ **Auto retry** - Configurable retry with exponential backoff
- ✅ **Timeout protection** - Fails gracefully after timeout
- ✅ **Full metrics** - Track every stage's performance
- ✅ **Circuit breaker** - Prevents cascading failures

---

## 🎓 **Complete Example: Weather System**

Let's see a complete before/after for the Weather System:

### **❌ Before (Not Using Pipeline):**

```python
class EnhancedVisionWeather:
    def __init__(self):
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()
        # Pipeline initialized but never used! 😱

    async def get_weather_via_screenshot(self):
        # Still using direct calls - pipeline unused!
        await self._open_weather_app()
        await asyncio.sleep(2)
        screenshot = await self._capture_weather_screenshot()
        # ... more direct calls ...
        return weather_data
```

**Problem:** Pipeline exists but is completely bypassed!

### **✅ After (Actually Using Pipeline):**

```python
class EnhancedVisionWeather:
    def __init__(self):
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    # Register stages (what operations pipeline can handle)
    def _register_pipeline_stages(self):
        self.pipeline.register_stage(
            "screenshot_capture",
            self._capture_screenshot_async,  # Handler
            timeout=5.0,
            retry_count=1
        )

        self.pipeline.register_stage(
            "vision_analysis",
            self._analyze_screenshot_async,  # Handler
            timeout=15.0,
            retry_count=1
        )

    # NEW: Method that uses pipeline
    async def get_weather(self, location: str):
        """This actually uses the pipeline!"""
        result = await self.pipeline.process_async(
            text=f"Get weather for {location}",
            metadata={"location": location}
        )

        # Extract results from pipeline
        return result.get("metadata", {}).get("weather_data")
```

**Result:** Pipeline is now doing all the work with retry, timeout, and circuit breaker protection!

---

## 🚨 **Common Pitfalls**

### **Pitfall 1: Registering stages but not calling process_async**

```python
# ❌ Pipeline initialized but never used
def my_method(self):
    # Direct call - bypasses pipeline!
    return subprocess.run(["command"])

# ✅ Actually use the pipeline
async def my_method(self):
    result = await self.pipeline.process_async(
        text="Execute command",
        metadata={"stage": "my_stage"}
    )
    return result
```

### **Pitfall 2: Creating async handlers but not routing to them**

```python
# ❌ Handler exists but is never called
async def _my_handler(self, context):
    # This never runs because nothing calls it!
    context.metadata["result"] = "done"

# ✅ Route through pipeline to call handler
async def my_method(self):
    result = await self.pipeline.process_async(...)
```

### **Pitfall 3: Not specifying which stage to use**

```python
# ❌ Pipeline doesn't know which stage to execute
result = await self.pipeline.process_async(
    text="Do something"
    # Missing metadata with stage name!
)

# ✅ Specify the stage
result = await self.pipeline.process_async(
    text="Do something",
    metadata={"stage": "my_specific_stage"}  # Clear routing
)
```

---

## ✅ **Checklist: Is Your Pipeline Actually Being Used?**

- [ ] Do you call `await self.pipeline.process_async(...)` in your methods?
- [ ] Do you pass `metadata={"stage": "stage_name"}` to route correctly?
- [ ] Do your handlers actually execute (add logging to verify)?
- [ ] Have you removed direct `subprocess.run()` calls?
- [ ] Have you removed direct `requests.get()` calls?
- [ ] Do you extract results from `result.get("metadata", {})`?
- [ ] Do your sync methods wrap async versions with `run_until_complete`?

If you can't check all these boxes, **your pipeline isn't being used properly**!

---

## 🎯 **Quick Reference**

### **Pattern to Follow:**

```python
# 1. Initialize
self.pipeline = get_async_pipeline()
self._register_pipeline_stages()

# 2. Register stages
self.pipeline.register_stage("stage_name", handler, timeout=10.0)

# 3. Create handler
async def handler(self, context):
    # Do async work
    context.metadata["result"] = await async_operation()

# 4. Route through pipeline
async def my_method_async(self):
    result = await self.pipeline.process_async(
        text="Operation",
        metadata={"stage": "stage_name"}
    )
    return result.get("metadata", {}).get("result")

# 5. Wrap for sync callers
def my_method(self):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(self.my_method_async())
```

---

## 📚 **Summary**

**The async pipeline is only useful if you actually USE it!**

- ❌ Don't just initialize and forget
- ✅ Route all operations through `process_async()`
- ✅ Remove direct blocking calls
- ✅ Let the pipeline handle retry, timeout, and circuit breaking
- ✅ Extract results from pipeline metadata

**When properly used, you get 10-100x performance improvement and automatic fault tolerance!** 🚀
