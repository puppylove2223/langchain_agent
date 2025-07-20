import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class WorkflowEnhancer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=2000
        )
    
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
        
    def load_workflow_session(self, session_id: str) -> Optional[Dict]:
        """Load workflow data from session"""
        session_path = Path("sessions") / session_id
        workflow_file = session_path / "workflow.json"
        
        if not workflow_file.exists():
            print(f"❌ Workflow file not found for session {session_id}")
            return None
            
        with open(workflow_file, 'r') as f:
            return json.load(f)
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for vision API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_workflow_completeness(self, workflow_data: Dict) -> Dict:
        """Analyze workflow completeness using comprehensive LLM evaluation"""
        steps = workflow_data.get("steps", [])
        session_metadata = {
            "session_id": workflow_data.get("session_id", "unknown"),
            "created_at": workflow_data.get("created_at", "unknown"),
            "total_steps": len(steps)
        }
        
        if not steps:
            return {
                "is_complete": False,
                "issues": ["No workflow steps found"],
                "suggestions": ["Re-run workflow capture to get steps"],
                "clarity_score": 0.0,
                "analysis_confidence": "high"
            }
        
        # Multi-phase LLM analysis for comprehensive evaluation
        analysis_prompt = f"""
        You are a workflow documentation expert analyzing a captured workflow for completeness and clarity.
        
        SESSION METADATA:
        {json.dumps(session_metadata, indent=2)}
        
        WORKFLOW STEPS:
        {json.dumps(steps, indent=2)}
        
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
            
            # Transform to expected format while preserving rich analysis
            return {
                "is_complete": analysis.get("is_complete", False),
                "clarity_score": analysis.get("clarity_score", 0.0),
                "analysis_confidence": analysis.get("analysis_confidence", "medium"),
                "workflow_type": analysis.get("workflow_type", "unknown"),
                "issues": self._extract_issues_from_analysis(analysis),
                "suggestions": self._extract_suggestions_from_analysis(analysis),
                "unclear_steps": self._extract_unclear_steps(analysis),
                "detailed_analysis": analysis  # Preserve full analysis
            }
            
        except Exception as e:
            print(f"Error analyzing workflow: {e}")
            return {
                "is_complete": False,
                "issues": [f"Analysis failed: {e}"],
                "suggestions": ["Manual review required"],
                "clarity_score": 0.0,
                "analysis_confidence": "low"
            }

    def _extract_issues_from_analysis(self, analysis: Dict) -> List[str]:
        """Extract a flat list of issues from comprehensive analysis"""
        issues = []
        
        quality_issues = analysis.get("quality_issues", {})
        for category, items in quality_issues.items():
            if items:
                issues.extend([f"{category}: {item}" for item in items])
        
        coherence = analysis.get("coherence_analysis", {})
        if coherence.get("apparent_gaps"):
            issues.extend([f"Missing step: {gap}" for gap in coherence["apparent_gaps"]])
        
        return issues if issues else ["No specific issues identified"]

    def _extract_suggestions_from_analysis(self, analysis: Dict) -> List[str]:
        """Extract actionable suggestions from comprehensive analysis"""
        suggestions = []
        
        improvements = analysis.get("improvement_opportunities", {})
        if improvements.get("critical_gaps"):
            suggestions.extend(improvements["critical_gaps"])
        if improvements.get("enhancement_areas"):
            suggestions.extend(improvements["enhancement_areas"])
        
        return suggestions if suggestions else ["Consider manual review of workflow completeness"]

    def _extract_unclear_steps(self, analysis: Dict) -> List[int]:
        """Extract step numbers that need clarification"""
        unclear_steps = []
        
        quality_issues = analysis.get("quality_issues", {})
        for category in ["unclear_actions", "weak_motivations"]:
            step_refs = quality_issues.get(category, [])
            for ref in step_refs:
                # Extract step numbers from references like "step 3" or "3"
                import re
                numbers = re.findall(r'\d+', str(ref))
                unclear_steps.extend([int(n) for n in numbers])
        
        return list(set(unclear_steps))  # Remove duplicates
    
    def generate_refinement_questions(self, workflow_data: Dict, analysis: Dict) -> List[str]:
        """Generate contextual refinement questions using LLM reasoning"""
        
        # Extract comprehensive context
        steps = workflow_data.get("steps", [])
        detailed_analysis = analysis.get("detailed_analysis", {})
        workflow_type = analysis.get("workflow_type", "unknown workflow")
        
        question_generation_prompt = f"""
        You are helping to refine workflow documentation by generating targeted questions.
        
        WORKFLOW CONTEXT:
        - Type: {workflow_type}
        - Total Steps: {len(steps)}
        - Completeness: {"Complete" if analysis.get("is_complete") else "Incomplete"}
        - Clarity Score: {analysis.get("clarity_score", 0):.1f}/1.0
        
        WORKFLOW STEPS:
        {json.dumps(steps, indent=2)}
        
        ANALYSIS FINDINGS:
        {json.dumps(detailed_analysis, indent=2)}
        
        TASK: Generate specific, contextual questions that would most improve this workflow documentation.
        
        QUESTION GENERATION PRINCIPLES:
        1. Focus on the most impactful gaps or unclear areas
        2. Ask questions that would help generalize the workflow for reuse
        3. Prioritize understanding motivations over specific details
        4. Consider the workflow type and typical challenges in that domain
        5. Avoid redundant questions - each should address a distinct improvement
        
        QUESTION CATEGORIES TO CONSIDER:
        - Motivation clarification for specific steps
        - Missing workflow context or prerequisites  
        - Unclear decision points or branching logic
        - Overall workflow objectives and success criteria
        - Integration points with other processes
        - Error handling or alternative paths
        
        Generate 3-7 prioritized questions in JSON format:
        {{
            "priority_questions": [
                {{
                    "question": "Specific question text",
                    "rationale": "Why this question is important",
                    "category": "motivation|context|logic|objectives|integration|errors",
                    "target_improvement": "What this question aims to improve"
                }}
            ],
            "question_strategy": "Overall approach for refinement based on workflow type and issues"
        }}
        
        Make questions conversational and specific to this workflow context.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=question_generation_prompt)])
            response_text = self._extract_json_from_response(response.content)
            question_data = json.loads(response_text)
            
            questions = []
            for q_obj in question_data.get("priority_questions", []):
                questions.append(q_obj["question"])
            
            return questions
            
        except Exception as e:
            print(f"Error generating refinement questions: {e}")
            # Fallback to simpler question generation
            return self._generate_fallback_questions(workflow_data, analysis)

    def _generate_fallback_questions(self, workflow_data: Dict, analysis: Dict) -> List[str]:
        """Generate basic questions when LLM-based generation fails"""
        questions = []
        steps = workflow_data.get("steps", [])
        
        # Basic questions based on analysis
        if not analysis.get("is_complete", True):
            questions.append("What was the overall goal you were trying to achieve in this workflow?")
            
            unclear_steps = analysis.get("unclear_steps", [])
            for step_num in unclear_steps[:3]:  # Limit to 3 steps
                step = next((s for s in steps if s["step_number"] == step_num), None)
                if step:
                    questions.append(f"For step {step_num} - can you explain the motivation behind: {step['action']}?")
        
        if not questions:
            questions.append("Is there any additional context that would help someone else follow this workflow?")
        
        return questions
    
    def enhance_workflow_with_context(self, workflow_data: Dict, additional_context: str) -> Dict:
        """Enhance workflow with additional user-provided context"""
        steps = workflow_data.get("steps", [])
        
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
    
    def visual_verification_questions(self, workflow_data: Dict, step_numbers: List[int] = None) -> List[Dict]:
        """Generate questions that require looking at specific screenshots"""
        steps = workflow_data.get("steps", [])
        questions = []
        
        target_steps = steps if step_numbers is None else [s for s in steps if s["step_number"] in step_numbers]
        
        for step in target_steps:
            if step.get("confidence_score", 1.0) < 0.8:
                image_path = step.get("screenshot_path", "")
                if Path(image_path).exists():
                    question = {
                        "step_number": step["step_number"],
                        "question": f"Looking at this screenshot, can you confirm: {step['action']} - and explain what motivated this action?",
                        "screenshot_path": image_path,
                        "current_interpretation": {
                            "action": step["action"],
                            "motivation": step["motivation"]
                        }
                    }
                    questions.append(question)
        
        return questions
    
    def interactive_refinement_session(self, session_id: str):
        """Run an interactive session to refine the workflow"""
        print(f"🔍 Starting workflow refinement for session: {session_id}")
        
        # Load workflow data
        workflow_data = self.load_workflow_session(session_id)
        if not workflow_data:
            return
        
        print(f"📊 Loaded workflow with {len(workflow_data.get('steps', []))} steps")
        
        # Analyze completeness
        analysis = self.analyze_workflow_completeness(workflow_data)
        
        print(f"\n📈 Workflow Analysis:")
        print(f"Complete: {analysis.get('is_complete', False)}")
        print(f"Clarity Score: {analysis.get('clarity_score', 0):.1f}/1.0")
        
        if analysis.get("issues"):
            print("\n⚠️  Issues found:")
            for issue in analysis["issues"]:
                print(f"  - {issue}")
        
        # Generate and ask refinement questions
        questions = self.generate_refinement_questions(workflow_data, analysis)
        
        if questions:
            print(f"\n❓ Refinement Questions ({len(questions)}):")
            responses = []
            
            for i, question in enumerate(questions, 1):
                print(f"\n{i}. {question}")
                response = input("Your answer: ")
                responses.append(response)
            
            # Enhance workflow with responses
            combined_context = "\n".join([f"Q: {q}\nA: {r}" for q, r in zip(questions, responses)])
            enhanced_workflow = self.enhance_workflow_with_context(workflow_data, combined_context)
            
            # Save enhanced workflow
            enhanced_file = Path("sessions") / session_id / "workflow_enhanced.json"
            with open(enhanced_file, 'w') as f:
                json.dump(enhanced_workflow, f, indent=2)
            
            print(f"\n✅ Enhanced workflow saved: {enhanced_file}")
        
        # Visual verification if needed
        visual_questions = self.visual_verification_questions(workflow_data)
        if visual_questions:
            print(f"\n👀 Visual Verification Questions ({len(visual_questions)}):")
            print("These questions require looking at specific screenshots...")
            
            for vq in visual_questions:
                print(f"\nStep {vq['step_number']}: {vq['question']}")
                print(f"Screenshot: {vq['screenshot_path']}")
                print(f"Current interpretation: {vq['current_interpretation']['action']}")
                
                verify = input("Is this interpretation correct? (y/n/provide correction): ").strip().lower()
                if verify == 'n' or (verify != 'y' and verify != ''):
                    correction = verify if verify not in ['y', 'n'] else input("Please provide the correct interpretation: ")
                    # You could enhance the workflow further with these corrections
                    print(f"Noted: {correction}")
        
        print(f"\n🎉 Workflow refinement completed for session {session_id}")

if __name__ == "__main__":
    enhancer = WorkflowEnhancer()
    
    # List available sessions
    sessions_dir = Path("sessions")
    if sessions_dir.exists():
        sessions = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
        if sessions:
            print("Available sessions:")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. {session}")
            
            choice = input("\nEnter session number or ID: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                session_id = sessions[int(choice) - 1]
            else:
                session_id = choice
            
            enhancer.interactive_refinement_session(session_id)
        else:
            print("No sessions found. Run the main workflow agent first.")
    else:
        print("Sessions directory not found. Run the main workflow agent first.") 