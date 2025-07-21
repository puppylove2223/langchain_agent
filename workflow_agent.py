import asyncio
import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated
import threading

import pyautogui
from PIL import Image
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Global control variables
workflow_control = {
    "transition_to_enhancement": False,
    "stop_workflow": False,
    "force_human_input": False
}

class WorkflowStep(BaseModel):
    step_number: int
    action: str
    motivation: str
    ui_elements: List[str]
    timestamp: str
    screenshot_path: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class WorkflowState(TypedDict):
    session_id: str
    screenshots: List[str]
    steps: List[WorkflowStep]
    current_step: int
    analysis_complete: bool
    needs_human_input: bool
    human_question: str
    continue_workflow: bool
    phase: str  # "capture" or "enhancement"
    enhancement_complete: bool

class WorkflowAgent:
    def __init__(self, human_interaction_level: str = "balanced"):
        """
        Initialize WorkflowAgent
        
        Args:
            human_interaction_level: "conservative", "balanced", or "frequent"
        """
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000
        )
        self.session_folder = Path("sessions")
        self.session_folder.mkdir(exist_ok=True)
        self.keyboard_listener = None
        self.human_interaction_level = human_interaction_level
        self._setup_keyboard_listener()
    
    def _setup_keyboard_listener(self):
        """Setup global keyboard listener for Ctrl+R and Ctrl+C"""
        try:
            from pynput import keyboard
            
            def on_key_combination():
                """Handle Ctrl+R combination"""
                print("\n🔄 Ctrl+R detected - Transitioning to enhancement phase...")
                workflow_control["transition_to_enhancement"] = True
            
            def on_stop_combination():
                """Handle Ctrl+Shift+Q combination to stop"""
                print("\n⏹️ Ctrl+Shift+Q detected - Stopping workflow...")
                workflow_control["stop_workflow"] = True
            
            def on_manual_question():
                """Handle Ctrl+H to manually trigger human input"""
                print("\n❓ Ctrl+H detected - Will ask for clarification on next step...")
                workflow_control["force_human_input"] = True
            
            # Set up hotkey combinations
            ctrl_r = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+r'), on_key_combination)
            ctrl_shift_q = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+<shift>+q'), on_stop_combination)
            ctrl_h = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+h'), on_manual_question)
            
            def for_canonical(f):
                return lambda k: f(self.keyboard_listener.canonical(k))
            
            hotkeys = [ctrl_r, ctrl_shift_q, ctrl_h]
            
            def on_press(key):
                for hotkey in hotkeys:
                    hotkey.press(self.keyboard_listener.canonical(key))
            
            def on_release(key):
                for hotkey in hotkeys:
                    hotkey.release(self.keyboard_listener.canonical(key))
            
            self.keyboard_listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            self.keyboard_listener.start()
            
            print("⌨️  Keyboard shortcuts activated:")
            print("   Ctrl+R: Transition to enhancement phase")  
            print("   Ctrl+H: Force human input question on next step")
            print("   Ctrl+Shift+Q: Stop workflow")
            print("   Ctrl+C: Emergency stop")
            
        except ImportError:
            print("⚠️  pynput not available. Install with: pip install pynput")
            print("   Falling back to Ctrl+C only for stopping")
        except Exception as e:
            print(f"⚠️  Could not setup keyboard listener: {e}")
            print("   Falling back to Ctrl+C only for stopping")

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks"""
        response_text = response_content.strip()
        if response_text.startswith("```json"):
            # Remove markdown formatting
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            # Handle generic code blocks
            response_text = response_text.replace("```", "").strip()
        return response_text
        
    def create_session(self) -> str:
        """Create a new session with unique ID"""
        session_id = str(uuid.uuid4())[:8]
        session_path = self.session_folder / session_id
        session_path.mkdir(exist_ok=True)
        (session_path / "screenshots").mkdir(exist_ok=True)
        return session_id

    def capture_screenshot(self, session_id: str, step_number: int) -> str:
        """Capture screenshot and save to session folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{step_number}_{timestamp}.png"
        filepath = self.session_folder / session_id / "screenshots" / filename
        
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        
        print(f"📸 Screenshot captured: {filename}")
        return str(filepath)

    def analyze_screenshot(self, screenshot_path: str, previous_steps: List[WorkflowStep]) -> Dict:
        """Analyze screenshot to determine UI changes and motivations using pure LLM reasoning"""
        
        # Create rich context from previous steps
        context = self._build_workflow_context(previous_steps)
        
        # First pass: Basic analysis
        analysis_prompt = f"""
        You are analyzing a screenshot as part of an ongoing workflow documentation process.
        
        WORKFLOW CONTEXT:
        {context}
        
        TASK: Analyze this screenshot to understand what happened and why.
        
        IMPORTANT GUIDELINES:
        1. Describe actions in GENERAL terms (e.g., "USER SELECTED FROM DROPDOWN" not "selected 'Marketing' from Department dropdown")
        2. Focus on the USER'S INTENT and MOTIVATION behind the action
        3. Consider the broader workflow context to infer purpose
        4. Be honest about uncertainty - if motivation is unclear from context, acknowledge it
        
        Analyze the screenshot and respond in JSON format:
        {{
            "action": "General description of the UI action that occurred",
            "motivation": "The likely reason/purpose behind this action based on context",
            "ui_elements": ["list", "of", "ui", "element", "types"],
            "workflow_progression": "How this step advances the overall workflow",
            "certainty_level": "high|medium|low",
            "analysis_notes": "Any observations about context, assumptions made, or areas of uncertainty"
        }}
        
        Be thorough in your reasoning but concise in your output.
        """
        
        try:
            # Create message with image for GPT-4o
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            
            # Extract JSON from markdown code blocks if present
            response_text = self._extract_json_from_response(response.content)
            basic_analysis = json.loads(response_text)
            
            # Check for manual trigger first
            if workflow_control["force_human_input"]:
                workflow_control["force_human_input"] = False  # Reset flag
                return {
                    "action": basic_analysis["action"],
                    "motivation": basic_analysis["motivation"],
                    "ui_elements": basic_analysis["ui_elements"],
                    "confidence_score": self._certainty_to_score(basic_analysis["certainty_level"]),
                    "needs_clarification": True,
                    "clarification_question": f"[Manual trigger] Can you provide more context about this step: {basic_analysis['action']}? What was your specific motivation?",
                    "workflow_progression": basic_analysis.get("workflow_progression", ""),
                    "analysis_notes": basic_analysis.get("analysis_notes", ""),
                    "clarification_focus": "motivation"
                }
            
            # Get interaction guidelines based on level
            interaction_guidelines = self._get_interaction_guidelines()
            
            # Second pass: Determine if clarification is needed
            clarification_prompt = f"""
            Review this workflow step analysis and determine if human clarification would improve understanding:
            
            ANALYSIS:
            {json.dumps(basic_analysis, indent=2)}
            
            WORKFLOW CONTEXT:
            {context}
            
            EVALUATION CRITERIA:
            1. Is the motivation clear and well-reasoned based on available context?
            2. Does the action make logical sense in the workflow progression?
            3. Are there multiple plausible interpretations of the user's intent?
            4. Would additional context significantly improve workflow documentation?
            
            Respond in JSON format:
            {{
                "needs_clarification": true/false,
                "confidence_assessment": "Explanation of why clarification is/isn't needed",
                "clarification_question": "Specific question to ask user (only if needs_clarification is true)",
                "clarification_focus": "What aspect needs clarification: motivation|action|context|purpose"
            }}
            
            {interaction_guidelines}
            """
            
            clarification_response = self.llm.invoke([HumanMessage(content=clarification_prompt)])
            
            # Extract JSON from markdown code blocks if present
            clarification_text = self._extract_json_from_response(clarification_response.content)
            clarification_analysis = json.loads(clarification_text)
            
            # Combine analyses
            return {
                "action": basic_analysis["action"],
                "motivation": basic_analysis["motivation"],
                "ui_elements": basic_analysis["ui_elements"],
                "confidence_score": self._certainty_to_score(basic_analysis["certainty_level"]),
                "needs_clarification": clarification_analysis["needs_clarification"],
                "clarification_question": clarification_analysis.get("clarification_question", ""),
                "workflow_progression": basic_analysis.get("workflow_progression", ""),
                "analysis_notes": basic_analysis.get("analysis_notes", ""),
                "clarification_focus": clarification_analysis.get("clarification_focus", "")
            }
            
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            # Even error handling uses LLM for contextual questions
            return self._generate_error_response(e, context)

    def _build_workflow_context(self, previous_steps: List[WorkflowStep]) -> str:
        """Build rich contextual information about the workflow so far"""
        if not previous_steps:
            return "This is the first step in the workflow - no previous context available."
        
        context_parts = []
        context_parts.append(f"WORKFLOW PROGRESS: {len(previous_steps)} steps completed so far")
        
        # Recent steps for immediate context
        if len(previous_steps) >= 3:
            context_parts.append("\nRECENT STEPS:")
            for step in previous_steps[-3:]:
                context_parts.append(f"  Step {step.step_number}: {step.action}")
                context_parts.append(f"    → Motivation: {step.motivation}")
        else:
            context_parts.append("\nALL PREVIOUS STEPS:")
            for step in previous_steps:
                context_parts.append(f"  Step {step.step_number}: {step.action}")
                context_parts.append(f"    → Motivation: {step.motivation}")
        
        # Workflow pattern recognition
        if len(previous_steps) >= 2:
            context_parts.append(f"\nWORKFLOW PATTERN OBSERVED:")
            context_parts.append(f"  The user appears to be engaged in: {self._infer_workflow_type(previous_steps)}")
        
        return "\n".join(context_parts)

    def _infer_workflow_type(self, steps: List[WorkflowStep]) -> str:
        """Use LLM to infer the type of workflow based on steps so far"""
        steps_summary = []
        for step in steps:
            steps_summary.append(f"Step {step.step_number}: {step.action} ({step.motivation})")
        
        inference_prompt = f"""
        Based on these workflow steps, what type of process is the user likely performing?
        
        STEPS:
        {chr(10).join(steps_summary)}
        
        Provide a brief, general description of the workflow type (e.g., "data entry process", "system configuration", "content creation workflow", "user account management", etc.)
        
        Respond with just the workflow type description, nothing else.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=inference_prompt)])
            return response.content.strip()
        except:
            return "general workflow process"

    def _certainty_to_score(self, certainty_level: str) -> float:
        """Convert certainty level to numeric score"""
        mapping = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.4
        }
        return mapping.get(certainty_level.lower(), 0.5)
    
    def _get_interaction_guidelines(self) -> str:
        """Get guidelines for human interaction based on configured level"""
        guidelines = {
            "conservative": """
            Only request clarification if it would genuinely improve workflow understanding.
            Be conservative - prefer reasonable assumptions over interrupting the user.
            """,
            "balanced": """
            Request clarification when:
            - Multiple reasonable interpretations exist
            - The motivation could be clearer for documentation purposes  
            - Context would significantly help future users understand the workflow
            - The action appears complex or non-obvious
            
            Balance thoroughness with user experience - err slightly toward asking questions for better documentation.
            """,
            "frequent": """
            Request clarification more actively to create comprehensive documentation:
            - When any aspect could benefit from additional context
            - To ensure motivations are clearly documented
            - When steps might be unclear to future users
            - To capture domain-specific knowledge
            
            Prioritize thorough documentation over minimal interruption.
            """
        }
        return guidelines.get(self.human_interaction_level, guidelines["balanced"])

    def _generate_error_response(self, error: Exception, context: str) -> Dict:
        """Generate contextual error response using LLM"""
        error_prompt = f"""
        Screenshot analysis failed with error: {str(error)}
        
        WORKFLOW CONTEXT:
        {context}
        
        Generate an appropriate question to ask the user to understand what happened in this step.
        Make the question specific to the workflow context and helpful for documentation.
        
        Respond in JSON format:
        {{
            "clarification_question": "Contextual question based on workflow and error"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=error_prompt)])
            response_text = self._extract_json_from_response(response.content)
            error_analysis = json.loads(response_text)
            return {
                "action": "ANALYSIS_ERROR_OCCURRED",
                "motivation": "Unable to automatically analyze this step",
                "ui_elements": [],
                "confidence_score": 0.0,
                "needs_clarification": True,
                "clarification_question": error_analysis.get("clarification_question", "Could you describe what action you took in this step?"),
                "analysis_notes": f"Technical error: {str(error)}"
            }
        except:
            return {
                "action": "ANALYSIS_ERROR_OCCURRED",
                "motivation": "Unable to automatically analyze this step",
                "ui_elements": [],
                "confidence_score": 0.0,
                "needs_clarification": True,
                "clarification_question": "Could you describe what action you took in this step and why?",
                "analysis_notes": f"Technical error: {str(error)}"
            }

    def save_workflow_data(self, session_id: str, steps: List[WorkflowStep]):
        """Save workflow steps to JSON file"""
        workflow_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "steps": [step.model_dump() for step in steps]
        }
        
        filepath = self.session_folder / session_id / "workflow.json"
        with open(filepath, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        print(f"💾 Workflow data saved: {filepath}")

    def ask_human(self, question: str) -> str:
        """Ask human for clarification"""
        print(f"\n🤔 Human input needed:")
        print(f"Question: {question}")
        response = input("Your answer: ")
        return response

    async def capture_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Capture screenshot every 10 seconds"""
        if not state["continue_workflow"]:
            return state
            
        # Wait 10 seconds before capture (except first time)
        if state["current_step"] > 0:
            print("⏳ Waiting 10 seconds before next capture...")
            await asyncio.sleep(10)
        
        # Capture screenshot
        screenshot_path = self.capture_screenshot(
            state["session_id"], 
            state["current_step"] + 1
        )
        
        state["screenshots"].append(screenshot_path)
        state["current_step"] += 1
        
        return state

    def analyze_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Analyze the latest screenshot"""
        if not state["screenshots"]:
            return state
            
        latest_screenshot = state["screenshots"][-1]
        
        # Analyze the screenshot
        analysis = self.analyze_screenshot(latest_screenshot, state["steps"])
        
        # Create workflow step
        step = WorkflowStep(
            step_number=state["current_step"],
            action=analysis["action"],
            motivation=analysis["motivation"],
            ui_elements=analysis["ui_elements"],
            timestamp=datetime.now().isoformat(),
            screenshot_path=latest_screenshot,
            confidence_score=analysis["confidence_score"]
        )
        
        # Check if human input is needed
        if analysis["needs_clarification"]:
            state["needs_human_input"] = True
            state["human_question"] = analysis["clarification_question"]
        else:
            state["steps"].append(step)
            state["needs_human_input"] = False
            
        return state

    def human_input_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Handle human-in-the-loop interaction with contextual LLM enhancement"""
        if not state["needs_human_input"]:
            return state
            
        # Ask human for clarification
        human_response = self.ask_human(state["human_question"])
        
        # Update the last step with human input using contextual LLM processing
        if state["screenshots"]:
            latest_screenshot = state["screenshots"][-1]
            
            # Build comprehensive context for enhancement
            workflow_context = self._build_workflow_context(state["steps"])
            
            # Use LLM to intelligently integrate human feedback
            integration_prompt = f"""
            You need to create a comprehensive workflow step based on human clarification.
            
            WORKFLOW CONTEXT:
            {workflow_context}
            
            HUMAN CLARIFICATION PROVIDED:
            "{human_response}"
            
            ORIGINAL QUESTION ASKED:
            "{state['human_question']}"
            
            TASK: 
            Create a complete workflow step that integrates the human's clarification with the visual context.
            Focus on extracting the general action pattern and the underlying motivation.
            
            GUIDELINES:
            1. Use the human's explanation to understand the TRUE motivation
            2. Generalize the action (avoid specific values/text)
            3. Infer UI elements that would be involved in this type of action
            4. Consider how this step fits into the broader workflow pattern
            
            Respond in JSON format:
            {{
                "action": "General action description based on human clarification",
                "motivation": "Clear motivation derived from human input",
                "ui_elements": ["inferred", "ui", "elements"],
                "workflow_integration": "How this step fits into the overall workflow",
                "confidence_rationale": "Why this interpretation is reliable based on human input"
            }}
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=integration_prompt)])
                response_text = self._extract_json_from_response(response.content)
                enhanced_analysis = json.loads(response_text)
                
                step = WorkflowStep(
                    step_number=state["current_step"],
                    action=enhanced_analysis.get("action", "USER_PROVIDED_ACTION"),
                    motivation=enhanced_analysis.get("motivation", human_response),
                    ui_elements=enhanced_analysis.get("ui_elements", []),
                    timestamp=datetime.now().isoformat(),
                    screenshot_path=latest_screenshot,
                    confidence_score=1.0  # High confidence with human input
                )
                
                state["steps"].append(step)
                
                # Optionally log the enhancement details
                print(f"🔄 Enhanced step with context: {enhanced_analysis.get('workflow_integration', '')}")
                
            except Exception as e:
                print(f"Error enhancing with human input: {e}")
                # Fallback with direct human input
                step = WorkflowStep(
                    step_number=state["current_step"],
                    action=self._extract_action_from_human_input(human_response),
                    motivation=human_response,
                    ui_elements=[],
                    timestamp=datetime.now().isoformat(),
                    screenshot_path=latest_screenshot,
                    confidence_score=1.0
                )
                state["steps"].append(step)
        
        state["needs_human_input"] = False
        state["human_question"] = ""
        
        return state

    def _extract_action_from_human_input(self, human_input: str) -> str:
        """Extract a general action description from human input using LLM"""
        extraction_prompt = f"""
        The user provided this explanation of their action: "{human_input}"
        
        Extract a general, reusable action description that could apply to similar workflows.
        Focus on the type of action, not specific details.
        
        Examples:
        - "I clicked the save button" → "USER_CLICKED_SAVE_BUTTON"
        - "I filled in my email address" → "USER_ENTERED_EMAIL_ADDRESS"
        - "I selected the marketing department" → "USER_SELECTED_DEPARTMENT_OPTION"
        
        Respond with just the general action description in ALL_CAPS_WITH_UNDERSCORES format.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            return response.content.strip()
        except:
            return "USER_PERFORMED_ACTION"

    def enhancement_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Analyze workflow completeness using comprehensive LLM evaluation"""
        print("🔍 Analyzing workflow completeness...")
        
        # Load workflow data
        workflow_data = {
            "session_id": state["session_id"],
            "created_at": datetime.now().isoformat(),
            "steps": [step.model_dump() for step in state["steps"]]
        }
        
        steps = state["steps"]
        session_metadata = {
            "session_id": state["session_id"],
            "total_steps": len(steps)
        }
        
        if not steps:
            print("❌ No workflow steps found")
            state["enhancement_complete"] = True
            return state
        
        # Multi-phase LLM analysis for comprehensive evaluation
        analysis_prompt = f"""
        You are a workflow documentation expert analyzing a captured workflow for completeness and clarity.
        
        SESSION METADATA:
        {json.dumps(session_metadata, indent=2)}
        
        WORKFLOW STEPS:
        {json.dumps([step.model_dump() for step in steps], indent=2)}
        
        COMPREHENSIVE EVALUATION FRAMEWORK:
        
        1. WORKFLOW COHERENCE:
           - Does the sequence of steps form a logical, complete process?
           - Are there apparent gaps or missing steps in the workflow?
           - Do the steps build toward a clear objective?
        
        2. ACTION CLARITY:
           - Are action descriptions sufficiently general yet descriptive?
           - Can actions be understood without seeing the screenshots?
           - Are actions consistently formatted and categorized?
        
        3. MOTIVATION DEPTH:
           - Are motivations clearly explained and contextually appropriate?
           - Do motivations help understand the "why" behind each action?
           - Are there steps where motivation seems unclear or insufficient?
        
        4. WORKFLOW GENERALIZABILITY:
           - Could this workflow documentation be applied to similar processes?
           - Are the descriptions domain-specific or appropriately general?
           - What patterns emerge that could apply to other workflows?
        
        5. DOCUMENTATION QUALITY:
           - What would make this workflow more useful as documentation?
           - Where would additional context significantly improve understanding?
           - What questions would someone following this workflow likely have?
        
        Provide a thorough analysis in JSON format:
        {{
            "overall_assessment": "comprehensive evaluation summary",
            "is_complete": true/false,
            "clarity_score": 0.0-1.0,
            "analysis_confidence": "high|medium|low",
            "workflow_type": "inferred type of workflow",
            "coherence_analysis": {{
                "logical_flow": "assessment of step sequence",
                "apparent_gaps": ["list of potential missing steps"],
                "objective_clarity": "how clear the overall goal is"
            }},
            "quality_issues": {{
                "unclear_actions": ["step numbers with unclear actions"],
                "weak_motivations": ["step numbers with insufficient motivation"],
                "inconsistent_formatting": ["formatting issues found"],
                "missing_context": ["areas needing more context"]
            }},
            "improvement_opportunities": {{
                "critical_gaps": ["most important missing information"],
                "enhancement_areas": ["areas that would benefit from clarification"],
                "generalization_needs": ["aspects that are too specific"]
            }},
            "suggested_questions": {{
                "motivation_clarifications": ["questions about unclear motivations"],
                "process_questions": ["questions about workflow logic"],
                "context_questions": ["questions about missing context"]
            }}
        }}
        
        Be thorough but practical in your assessment. Focus on actionable improvements.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            response_text = self._extract_json_from_response(response.content)
            analysis = json.loads(response_text)
            
            print(f"📈 Analysis Results:")
            print(f"Complete: {analysis.get('is_complete', False)}")
            print(f"Clarity Score: {analysis.get('clarity_score', 0):.1f}/1.0")
            print(f"Workflow Type: {analysis.get('workflow_type', 'unknown')}")
            
            # Store analysis in state for potential refinement
            state["enhancement_analysis"] = analysis
            
            # Check if refinement questions needed
            quality_issues = analysis.get("quality_issues", {})
            has_issues = any(quality_issues.values())
            
            if has_issues:
                state["needs_human_input"] = True
                state["human_question"] = "enhancement_refinement"
            else:
                state["enhancement_complete"] = True
                print("✅ Workflow analysis complete - no major issues found")
            
        except Exception as e:
            print(f"Error analyzing workflow: {e}")
            state["enhancement_complete"] = True
            
        return state

    def enhancement_refinement_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Generate and handle refinement questions"""
        if not state.get("needs_human_input") or state.get("human_question") != "enhancement_refinement":
            return state
            
        analysis = state.get("enhancement_analysis", {})
        
        # Generate contextual refinement questions
        print("\n❓ Generating refinement questions...")
        
        workflow_type = analysis.get("workflow_type", "unknown workflow")
        steps = state["steps"]
        
        question_generation_prompt = f"""
        You are helping to refine workflow documentation by generating targeted questions.
        
        WORKFLOW CONTEXT:
        - Type: {workflow_type}
        - Total Steps: {len(steps)}
        - Completeness: {"Complete" if analysis.get("is_complete") else "Incomplete"}
        - Clarity Score: {analysis.get("clarity_score", 0):.1f}/1.0
        
        ANALYSIS FINDINGS:
        {json.dumps(analysis, indent=2)}
        
        TASK: Generate 3-5 specific questions that would most improve this workflow documentation.
        
        QUESTION GENERATION PRINCIPLES:
        1. Focus on the most impactful gaps or unclear areas
        2. Ask questions that would help generalize the workflow for reuse
        3. Prioritize understanding motivations over specific details
        4. Consider the workflow type and typical challenges in that domain
        
        Generate prioritized questions in JSON format:
        {{
            "priority_questions": [
                "Question 1 text",
                "Question 2 text", 
                "Question 3 text"
            ],
            "question_strategy": "Overall approach for refinement"
        }}
        
        Make questions conversational and specific to this workflow context.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=question_generation_prompt)])
            response_text = self._extract_json_from_response(response.content)
            question_data = json.loads(response_text)
            
            questions = question_data.get("priority_questions", [])
            
            if questions:
                print(f"\n🎯 Refinement Questions ({len(questions)}):")
                responses = []
                
                for i, question in enumerate(questions, 1):
                    print(f"\n{i}. {question}")
                    response = input("Your answer (or press Enter to skip): ").strip()
                    if response:
                        responses.append(response)
                
                if responses:
                    # Enhance workflow with responses
                    combined_context = "\n".join([f"Q: {q}\nA: {r}" for q, r in zip(questions, responses) if r])
                    enhanced_workflow = self._enhance_workflow_with_context(state, combined_context)
                    
                    # Save enhanced workflow
                    enhanced_file = self.session_folder / state["session_id"] / "workflow_enhanced.json"
                    with open(enhanced_file, 'w') as f:
                        json.dump(enhanced_workflow, f, indent=2)
                    
                    print(f"\n✅ Enhanced workflow saved: {enhanced_file}")
                else:
                    print("\n⏭️  No responses provided - skipping enhancement")
            
        except Exception as e:
            print(f"Error generating refinement questions: {e}")
        
        state["needs_human_input"] = False
        state["enhancement_complete"] = True
        return state

    def _enhance_workflow_with_context(self, state: WorkflowState, additional_context: str) -> Dict:
        """Enhance workflow with additional user-provided context"""
        workflow_data = {
            "session_id": state["session_id"],
            "created_at": datetime.now().isoformat(),
            "steps": [step.model_dump() for step in state["steps"]]
        }
        
        steps = [step.model_dump() for step in state["steps"]]
        
        enhancement_prompt = f"""
        Original workflow steps:
        {json.dumps(steps, indent=2)}
        
        Additional context provided by user:
        {additional_context}
        
        Based on this additional context, enhance the workflow by:
        1. Updating motivations where they were unclear
        2. Adding missing context to actions
        3. Improving the overall flow description
        4. Filling in any logical gaps
        
        Return the enhanced workflow in the same JSON format but with improved descriptions.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=enhancement_prompt)])
            response_text = self._extract_json_from_response(response.content)
            enhanced_data = json.loads(response_text)
            
            # Update the original workflow data
            enhanced_workflow = workflow_data.copy()
            enhanced_workflow["steps"] = enhanced_data
            enhanced_workflow["enhanced_at"] = datetime.now().isoformat()
            enhanced_workflow["enhancement_context"] = additional_context
            
            return enhanced_workflow
            
        except Exception as e:
            print(f"Error enhancing workflow: {e}")
            return workflow_data

    def check_continuation_node(self, state: WorkflowState) -> WorkflowState:
        """Node: Check workflow status and handle phase transitions"""
        # Check for stop signal
        if workflow_control["stop_workflow"]:
            print(f"\n🛑 Workflow stopped by user")
            state["continue_workflow"] = False
            state["analysis_complete"] = True
            self.save_workflow_data(state["session_id"], state["steps"])
            return state
            
        # Check for phase transition
        if workflow_control["transition_to_enhancement"] and state["phase"] == "capture":
            print(f"\n🔄 Transitioning to enhancement phase...")
            state["phase"] = "enhancement"
            state["continue_workflow"] = False  # Stop capture loop
            state["analysis_complete"] = False   # Continue to enhancement
            workflow_control["transition_to_enhancement"] = False  # Reset flag
            self.save_workflow_data(state["session_id"], state["steps"])
            return state
        
        # Display current progress
        print(f"\n📊 Workflow Progress [{state['phase'].upper()} PHASE]:")
        print(f"Session ID: {state['session_id']}")
        print(f"Steps captured: {len(state['steps'])}")
        
        if state["steps"]:
            print("Latest step:")
            latest = state["steps"][-1]
            print(f"  Action: {latest.action}")
            print(f"  Motivation: {latest.motivation}")
        
        print(f"⌨️  Press Ctrl+H to ask question, Ctrl+R for enhancement phase, Ctrl+Shift+Q to stop")
        
        return state

    def create_workflow_graph(self) -> StateGraph:
        """Create the unified LangGraph workflow with capture and enhancement phases"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("capture", self.capture_workflow_node)
        workflow.add_node("analyze", self.analyze_workflow_node)
        workflow.add_node("human_input", self.human_input_node)
        workflow.add_node("check_continuation", self.check_continuation_node)
        workflow.add_node("enhancement_analysis", self.enhancement_analysis_node)
        workflow.add_node("enhancement_refinement", self.enhancement_refinement_node)
        
        # Set entry point
        workflow.set_entry_point("capture")
        
        # Capture phase edges
        workflow.add_edge("capture", "analyze")
        
        workflow.add_conditional_edges(
            "analyze",
            lambda state: "human_input" if state["needs_human_input"] else "check_continuation",
            {"human_input": "human_input", "check_continuation": "check_continuation"}
        )
        
        workflow.add_edge("human_input", "check_continuation")
        
        # Phase transition logic
        workflow.add_conditional_edges(
            "check_continuation",
            lambda state: (
                END if workflow_control["stop_workflow"] or state["analysis_complete"] else
                "enhancement_analysis" if state["phase"] == "enhancement" else
                "capture"
            ),
            {
                "capture": "capture", 
                "enhancement_analysis": "enhancement_analysis",
                END: END
            }
        )
        
        # Enhancement phase edges
        workflow.add_conditional_edges(
            "enhancement_analysis",
            lambda state: "enhancement_refinement" if state["needs_human_input"] else END,
            {"enhancement_refinement": "enhancement_refinement", END: END}
        )
        
        workflow.add_edge("enhancement_refinement", END)
        
        return workflow.compile()

    async def run_workflow(self):
        """Main method to run the unified workflow"""
        print("🚀 Starting Unified LangGraph Workflow Agent")
        print("This agent captures screenshots, analyzes workflow, and provides enhancement.")
        print("Use Ctrl+R to transition phases, Ctrl+Shift+Q to stop.\n")
        
        # Create session
        session_id = self.create_session()
        print(f"📁 Session created: {session_id}")
        
        # Initialize state
        initial_state: WorkflowState = {
            "session_id": session_id,
            "screenshots": [],
            "steps": [],
            "current_step": 0,
            "analysis_complete": False,
            "needs_human_input": False,
            "human_question": "",
            "continue_workflow": True,
            "phase": "capture",
            "enhancement_complete": False
        }
        
        # Create and run workflow
        workflow = self.create_workflow_graph()
        
        try:
            final_state = await workflow.ainvoke(
                initial_state,
                config={"recursion_limit": 50}
            )
            
            print(f"\n✅ Complete workflow finished!")
            print(f"📁 Session folder: {self.session_folder / session_id}")
            print(f"📊 Total steps captured: {len(final_state['steps'])}")
            print(f"📸 Screenshots captured: {len(final_state['screenshots'])}")
            
            # Check for enhanced workflow
            enhanced_file = self.session_folder / session_id / "workflow_enhanced.json"
            if enhanced_file.exists():
                print(f"✨ Enhanced workflow available: {enhanced_file}")
            
            return final_state
            
        except KeyboardInterrupt:
            print("\n⏹️  Workflow interrupted by user (Ctrl+C)")
            self.save_workflow_data(session_id, initial_state["steps"])
            return initial_state
        finally:
            # Clean up keyboard listener
            if self.keyboard_listener:
                self.keyboard_listener.stop()

if __name__ == "__main__":
    # You can configure human interaction level:
    # "conservative" - Rarely asks questions (original behavior)
    # "balanced" - Moderate interaction (default, improved)  
    # "frequent" - More active questioning for thorough documentation
    
    agent = WorkflowAgent(human_interaction_level="balanced")
    asyncio.run(agent.run_workflow()) 