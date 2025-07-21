# LangGraph Workflow Agent - Code Index

## 📁 Project Structure

```
LangGraph Agent/
├── workflow_agent.py          # Main LangGraph workflow agent (992 lines)
├── enhanced_analyzer.py       # Post-processing workflow enhancement (447 lines)
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore patterns
├── sessions/                # Session storage directory
│   ├── [session_id]/       # Individual session folders
│   │   ├── screenshots/    # Captured screenshots
│   │   ├── workflow.json   # Workflow data
│   │   └── workflow_enhanced.json # Enhanced workflow data
└── venv/                   # Virtual environment
```

## 🏗️ Core Components

### 1. `workflow_agent.py` - Main Workflow Agent

#### **Classes & Data Models**

**`WorkflowStep` (BaseModel)**
- `step_number`: Sequential step identifier
- `action`: General description of UI action
- `motivation`: User intent behind the action
- `ui_elements`: List of UI element types involved
- `timestamp`: When the step occurred
- `screenshot_path`: Path to captured screenshot
- `confidence_score`: LLM confidence (0.0-1.0)

**`WorkflowState` (TypedDict)**
- `session_id`: Unique session identifier
- `screenshots`: List of screenshot file paths
- `steps`: List of WorkflowStep objects
- `current_step`: Current step counter
- `analysis_complete`: Whether analysis is finished
- `needs_human_input`: Whether clarification is needed
- `human_question`: Question to ask user
- `continue_workflow`: Whether to continue processing
- `phase`: Current phase ("capture" or "enhancement")
- `enhancement_complete`: Whether enhancement is finished

**`WorkflowAgent` (Main Class)**
- **Initialization**: Sets up LLM, session management, keyboard listeners
- **Human Interaction Levels**: "conservative", "balanced", "frequent"

#### **Key Methods**

**Session Management:**
- `create_session()`: Creates unique session with UUID
- `capture_screenshot()`: Takes screenshots with timestamps
- `save_workflow_data()`: Saves workflow to JSON

**Analysis & Processing:**
- `analyze_screenshot()`: LLM-driven screenshot analysis
- `_build_workflow_context()`: Creates rich context from previous steps
- `_infer_workflow_type()`: Identifies workflow patterns
- `_get_interaction_guidelines()`: Determines human interaction frequency

**LangGraph Nodes:**
- `capture_workflow_node()`: Screenshot capture and analysis
- `analyze_workflow_node()`: LLM analysis of captured data
- `human_input_node()`: Handles user clarification requests
- `enhancement_analysis_node()`: Comprehensive workflow enhancement
- `enhancement_refinement_node()`: Refines enhanced workflows
- `check_continuation_node()`: Determines workflow continuation

**Workflow Control:**
- `create_workflow_graph()`: Builds LangGraph state machine
- `run_workflow()`: Main execution loop

#### **Keyboard Controls**
- `Ctrl+R`: Transition to enhancement phase
- `Ctrl+Shift+Q`: Stop workflow gracefully
- `Ctrl+H`: Manually trigger human input
- `Ctrl+C`: Emergency stop

### 2. `enhanced_analyzer.py` - Post-Processing Enhancement

#### **Classes**

**`WorkflowEnhancer`**
- **Purpose**: Post-processing analysis and enhancement of captured workflows
- **LLM Integration**: Uses GPT-4o for comprehensive analysis

#### **Key Methods**

**Analysis & Evaluation:**
- `analyze_workflow_completeness()`: Comprehensive workflow evaluation
- `_extract_issues_from_analysis()`: Identifies workflow problems
- `_extract_suggestions_from_analysis()`: Generates improvement suggestions
- `_extract_unclear_steps()`: Finds steps needing clarification

**Enhancement & Refinement:**
- `generate_refinement_questions()`: Creates contextual questions
- `enhance_workflow_with_context()`: Adds context to workflows
- `visual_verification_questions()`: Generates verification questions
- `interactive_refinement_session()`: Interactive enhancement session

**Utility Methods:**
- `load_workflow_session()`: Loads session data
- `encode_image()`: Base64 encoding for vision API
- `_extract_json_from_response()`: Parses LLM responses

## 🔄 Workflow Architecture

### **LangGraph State Machine**

```
Initial State
    ↓
Capture Node (screenshots + analysis)
    ↓
Analyze Node (LLM analysis)
    ↓
Human Input Node (if clarification needed)
    ↓
Check Continuation Node
    ↓
Enhancement Analysis Node (if phase transition)
    ↓
Enhancement Refinement Node
    ↓
End State
```

### **Data Flow**

1. **Capture Phase**:
   - Screenshots every 10 seconds
   - LLM analysis with workflow context
   - Human clarification when needed
   - JSON data persistence

2. **Enhancement Phase**:
   - Comprehensive workflow analysis
   - Pattern identification
   - Contextual refinement
   - Enhanced documentation generation

## 🧠 LLM Integration

### **Models Used**
- **GPT-4o**: Primary analysis and reasoning
- **Vision API**: Screenshot analysis
- **Temperature**: 0.1 for analysis, 0.3 for enhancement

### **Analysis Prompts**
- **Screenshot Analysis**: Context-aware UI action recognition
- **Workflow Context**: Rich context building from previous steps
- **Clarification Assessment**: Intelligent human-in-loop decisions
- **Enhancement Analysis**: Comprehensive workflow evaluation with repetitive pattern detection
- **Pattern Consolidation**: Automatic identification and generalization of repetitive steps

## 📊 Session Management

### **Session Structure**
```
sessions/[session_id]/
├── screenshots/
│   ├── screenshot_1_20241201_143022.png
│   ├── screenshot_2_20241201_143032.png
│   └── ...
├── workflow.json
└── workflow_enhanced.json
```

### **Data Formats**
- **workflow.json**: Raw captured workflow data
- **workflow_enhanced.json**: Enhanced and refined workflow data

## 🔧 Configuration & Environment

### **Environment Variables**
- `OPENAI_API_KEY`: Required for GPT-4 Vision analysis
- `LANGCHAIN_API_KEY`: Optional LangChain integration

### **Dependencies**
- `langgraph>=0.2.0`: LangGraph framework
- `langchain>=0.2.0`: LangChain integration
- `langchain-openai>=0.1.0`: OpenAI integration
- `pyautogui>=0.9.54`: Screenshot capture
- `pynput>=1.7.6`: Keyboard monitoring
- `pillow>=10.0.0`: Image processing
- `pydantic>=2.0.0`: Data validation

## 🎯 Key Features

### **LLM-Driven Analysis**
- 100% LLM-based decision making
- No rule-based logic
- Contextual intelligence
- Universal workflow support
- **Repetitive pattern detection and consolidation**

### **Human-in-the-Loop**
- Intelligent clarification requests
- Contextual question generation
- Conservative interruption strategy
- Manual trigger support

### **Workflow Generalization**
- Pattern recognition
- Reusable workflow templates
- Domain-agnostic analysis
- Contextual adaptation

## 🚀 Usage Patterns

### **Starting a New Session**
```python
agent = WorkflowAgent(human_interaction_level="balanced")
asyncio.run(agent.run_workflow())
```

### **Post-Processing Enhancement**
```python
enhancer = WorkflowEnhancer()
enhancer.interactive_refinement_session(session_id)
```

### **Session Analysis**
```python
workflow_data = enhancer.load_workflow_session(session_id)
analysis = enhancer.analyze_workflow_completeness(workflow_data)
```

## 🔍 Error Handling

### **Robust Error Management**
- JSON parsing fallbacks
- Screenshot capture error handling
- LLM response validation
- Graceful degradation

### **Recovery Mechanisms**
- Session data persistence
- Partial workflow recovery
- Error logging and reporting
- User notification system

## 📈 Performance Considerations

### **Optimization Strategies**
- Asynchronous processing
- Efficient screenshot storage
- LLM token optimization
- Memory management

### **Scalability Features**
- Session isolation
- Modular architecture
- Configurable parameters
- Extensible design 