# LangGraph Workflow Agent

A prototype LangGraph agent that captures screenshots at 10-second intervals, analyzes UI changes to generate workflow steps with motivations, and includes **contextual human-in-the-loop functionality** for unclear steps. **Fully LLM-driven analysis** ensures generalizability across any workflow type.

## 🎯 Key Design Philosophy

- **100% LLM-Driven**: No rule-based logic - all analysis, question generation, and refinement is contextually generated through LLM reasoning
- **Universal Workflow Support**: Designed to generalize across any workflow type (data entry, system configuration, content creation, etc.)
- **Contextual Intelligence**: Each decision is made based on comprehensive workflow context, not predetermined rules
- **Conservative Human-in-Loop**: Only interrupts when genuinely uncertain, using contextual analysis to determine necessity

## Features

- 📸 **Automatic Screenshot Capture**: Takes screenshots every 10 seconds
- 🧠 **Contextual AI Analysis**: Uses GPT-4o with comprehensive workflow context analysis
- 🤔 **Intelligent Human-in-the-Loop**: LLM determines when clarification would improve documentation
- 📊 **Session Management**: Organizes captures by unique session IDs
- 🔍 **Contextual Workflow Enhancement**: LLM-driven secondary analysis for refining workflows
- 💾 **JSON Export**: Saves workflow data for further processing
- 🎯 **Workflow Generalization**: Translates specific actions into reusable patterns

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### 3. Run the Workflow Agent

```bash
python workflow_agent.py
```

### 4. Enhance Captured Workflows

```bash
python enhanced_analyzer.py
```

## How It Works

### LLM-Driven Workflow Analysis

The agent uses a **multi-phase LLM analysis** approach:

1. **Screenshot Capture**: Agent takes screenshots every 10 seconds
2. **Contextual Analysis**: GPT-4o analyzes each screenshot with rich workflow context:
   - Previous steps and their motivations
   - Inferred workflow type and patterns
   - Overall workflow progression
   - UI element identification
3. **Clarification Assessment**: Separate LLM evaluation determines if human input would improve understanding:
   - Evaluates motivation clarity
   - Assesses workflow logic completeness
   - Considers multiple interpretation possibilities
   - Focuses on documentation improvement potential
4. **Contextual Questions**: If clarification needed, generates specific questions based on:
   - Workflow context and type
   - Specific areas of uncertainty
   - Overall documentation goals
5. **Intelligent Integration**: Human responses are contextually integrated using LLM reasoning

### Universal Workflow Patterns

The agent automatically identifies and adapts to different workflow types:
- **Data Entry Processes**: Form filling, spreadsheet work, database entry
- **System Configuration**: Settings adjustment, user management, system setup  
- **Content Creation**: Document editing, media creation, publishing workflows
- **Application Navigation**: Feature exploration, process execution, task completion

### LangGraph Workflow Structure

```
[Capture] → [Contextual Analysis] → [Clarification Assessment] → [Human Input?] → [Continue?]
    ↑              ↓                        ↓                        ↓           ↓
    ←──────────[Continue]────────────────────────────────────────────────────────
                                                                    ↓
                                                                  [END]
```

### Session Structure

```
sessions/
├── {session-id}/
│   ├── screenshots/
│   │   ├── screenshot_1_20241201_143022.png
│   │   ├── screenshot_2_20241201_143032.png
│   │   └── ...
│   ├── workflow.json
│   └── workflow_enhanced.json (after enhancement)
```

## Key Components

### WorkflowAgent (`workflow_agent.py`)

**Contextual Analysis Engine** that:
- Captures screenshots using `pyautogui`
- Performs multi-phase GPT-4o analysis with workflow context
- Uses LLM to determine when clarification is needed
- Contextually integrates human feedback
- Automatically infers workflow types and patterns

### WorkflowEnhancer (`enhanced_analyzer.py`)

**LLM-Driven Refinement Tool** that:
- Performs comprehensive workflow analysis using LLM evaluation frameworks
- Generates contextual refinement questions based on workflow type and issues
- Provides workflow generalizability assessment
- Creates domain-specific improvement suggestions

### Contextual Data Models

```python
class WorkflowStep:
    step_number: int
    action: str                # General action pattern
    motivation: str            # Contextually determined motivation
    ui_elements: List[str]     # UI components involved
    timestamp: str             # When captured
    screenshot_path: str       # Path to screenshot
    confidence_score: float    # LLM-assessed confidence
    workflow_progression: str  # How step advances workflow
    analysis_notes: str        # LLM reasoning notes
```

## LLM-Driven Features

### Contextual Question Generation

Questions are generated based on:
- **Workflow Type**: Domain-specific considerations (e.g., data entry vs. system configuration)
- **Context Analysis**: What information would most improve documentation
- **Pattern Recognition**: Common gaps in similar workflow types
- **Uncertainty Assessment**: Specific areas where LLM analysis was less confident

### Example Contextual Questions

**Data Entry Workflow**:
- "What validation rules or data requirements guided your input choices?"
- "Are there specific business rules that determine when this form should be used?"

**System Configuration**:
- "What outcome or system behavior were you trying to achieve with these settings?"
- "Are there dependencies or prerequisites that users should know about?"

**Content Creation**:
- "What content strategy or guidelines influenced these formatting decisions?"
- "How does this content fit into your overall publication workflow?"

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for GPT-4 Vision analysis
- `LANGCHAIN_API_KEY`: Optional, for LangChain tracing
- `LANGCHAIN_TRACING_V2`: Set to `true` for debugging
- `LANGCHAIN_PROJECT`: Project name for tracing

### LLM Customization Options

**Analysis Models**: Change GPT-4o model in both classes

**Context Depth**: Modify `_build_workflow_context()` for different context levels

**Clarification Sensitivity**: Adjust LLM prompts for more/less conservative questioning

**Workflow Types**: Extend workflow type inference in `_infer_workflow_type()`

## Generalizability Features

### Universal Action Patterns

- **INPUT_ACTIONS**: "USER_TYPED_IN_FIELD", "USER_SELECTED_FROM_DROPDOWN"
- **NAVIGATION_ACTIONS**: "USER_OPENED_APPLICATION", "USER_SWITCHED_TABS"  
- **INTERACTION_ACTIONS**: "USER_CLICKED_BUTTON", "USER_UPLOADED_FILE"
- **COMPLETION_ACTIONS**: "USER_SAVED_WORK", "USER_SUBMITTED_FORM"

### Cross-Domain Motivations

- **Data Collection**: "Gathering required information for process completion"
- **Validation**: "Ensuring data meets quality standards before submission"
- **Configuration**: "Setting up system parameters for desired behavior"
- **Review**: "Verifying information accuracy before finalizing"

## Limitations & Notes

- **Prototype Code**: Not production-ready, optimized for experimentation
- **LLM Dependencies**: Requires reliable OpenAI API access
- **Context Windows**: Very long workflows may hit token limits
- **Screen Dependencies**: Requires active screen for `pyautogui`
- **Vision API Costs**: GPT-4o multimodal calls can be expensive with many screenshots

## Sample Output

### workflow.json
```json
{
  "session_id": "a1b2c3d4",
  "created_at": "2024-12-01T14:30:22",
  "steps": [
    {
      "step_number": 1,
      "action": "USER_OPENED_APPLICATION",
      "motivation": "Initiating data entry workflow for customer information processing",
      "ui_elements": ["application_window", "menu_bar", "toolbar"],
      "workflow_progression": "Beginning of structured data collection process",
      "timestamp": "2024-12-01T14:30:22",
      "screenshot_path": "sessions/a1b2c3d4/screenshots/screenshot_1_20241201_143022.png",
      "confidence_score": 0.9,
      "analysis_notes": "Clear application launch detected with data entry interface"
    }
  ]
}
```

## Why LLM-Driven Matters

1. **Adaptability**: Works across any domain without modification
2. **Context Awareness**: Understands workflow patterns and makes informed decisions
3. **Quality Questions**: Generates meaningful clarifications specific to each situation
4. **Reduced Noise**: Conservative approach minimizes unnecessary interruptions
5. **Future-Proof**: Adapts to new workflow types automatically through LLM reasoning

This approach ensures the agent can handle everything from simple form filling to complex multi-application workflows without requiring domain-specific rules or modifications. 