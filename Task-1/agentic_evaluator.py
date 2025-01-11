import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
from langchain_groq import ChatGroq

@dataclass
class SectionAnalysis:
    content: str
    strengths: List[str]
    weaknesses: List[str]
    contribution_score: str  # 0-1 score of how much this section contributes to publishability
    reasoning: str

@dataclass
class PaperState:
    paper_id: str
    features: Dict[str, str]
    section_analyses: Dict[str, SectionAnalysis] = field(default_factory=dict)
    final_decision: str = ""
    reasoning: str = ""
    confidence: float = 0.0

@dataclass
class SystemState:
    papers_queue: List[PaperState]
    processed_papers: List[PaperState]
    memory: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

class BaseAnalysisAgent:
    """Base class for section analysis agents with ReACT reasoning"""
    
    def __init__(self, section_type):
        self.section_type = section_type
        self.llm = ChatGroq(
            temperature=0,
            api_key="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",
            model_name="llama-3.1-70b-versatile"
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert reviewer analyzing the {section_type} section of academic papers.
            Remember that publishable papers don't need to be perfect - they need to meet acceptable standards 
            and make meaningful contributions. Consider the following carefully:
            
            - A paper can be publishable even with some minor weaknesses if its strengths are significant
            - Focus on the substantive contribution rather than superficial issues
            - Consider the potential impact and novelty alongside methodological rigor
            - Look for evidence of sound scientific practice rather than perfection
            
            Follow these steps:
            1) Thought: Think about what key aspects to analyze in this {section_type} section
            2) Action: Analyze the content for these aspects, considering both strengths and limitations
            3) Observation: List specific observations, being careful to acknowledge positive aspects
            4) Thought: Evaluate how this section contributes to paper publishability, focusing on core scientific merit
            5) Action: Score the section's contribution (0-1) with balanced reasoning
            
            Previous similar papers in this category have shown these patterns:
            {patterns}
            
            Format your response as:
            Thought: [Your initial analysis approach]
            Action: [Your analysis steps]
            Observation: [Your findings]
            Thought: [Your evaluation]
            Action: [Final scoring]
            
            Final output must be in the format:
            STRENGTHS: [list of strengths]
            WEAKNESSES: [list of weaknesses]
            SCORE: [0-1 score. Should strictly be a float value]
            REASONING: [detailed reasoning]"""),
            ("human", "Content to analyze: {content}")
        ])
        
        self.chain = self.analysis_prompt | self.llm | StrOutputParser()
    
    def analyze(self, content: str, patterns: List[Dict[str, Any]]) -> SectionAnalysis:
        # Format patterns for prompt
        patterns_text = self._format_patterns(patterns)
        
        # Get analysis
        result = self.chain.invoke({
            "section_type": self.section_type,
            "content": content,
            "patterns": patterns_text
        })
        
        # Parse results
        strengths = self._extract_list(result, "STRENGTHS")
        weaknesses = self._extract_list(result, "WEAKNESSES")
        score = self._extract_value(result, "SCORE")
        reasoning = self._extract_value(result, "REASONING")
        
        return SectionAnalysis(
            content=content,
            strengths=strengths,
            weaknesses=weaknesses,
            contribution_score=score,
            reasoning=reasoning
        )
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        if not patterns:
            return "No previous patterns available."
        
        formatted = []
        for p in patterns[-3:]:  # Use last 3 patterns
            formatted.append(
                f"- Score: {p['contribution_score']}, "
                f"Decision: {p['decision']}, "
                f"Key points: {p['reasoning'][:100]}..."
            )
        return "\n".join(formatted)
    
    def _extract_list(self, text: str, marker: str) -> List[str]:
        try:
            section = text.split(f"{marker}:")[1].split("\n")[0]
            items = [item.strip() for item in section.split(",")]
            return [item for item in items if item]
        except:
            return []
    
    def _extract_value(self, text: str, marker: str) -> str:
        try:
            return text.split(f"{marker}:")[1].split("\n")[0].strip()
        except:
            return ""


class MethodologyAgent(BaseAnalysisAgent):
    def __init__(self):
        super().__init__("methodology")
        
        # self.llm = ChatGroq(
        #     temperature=0,
        #     api_key="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",
        #     model_name="mixtral-8x7b-32768"
        # )
        
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a methodology expert evaluating research papers for critical issues that affect publishability.
            Your primary focus is ensuring methodological appropriateness and justification.

            Critical Issues to Identify:
            1. Methodology Appropriateness:
               - Are the chosen methods well-suited to the research problem?
               - Is there proper justification for methodology selection?
               - Have techniques been appropriately adapted to the research context?
               - Are there any fundamental flaws in the methodological approach?

            2. Implementation Quality:
               - Is the methodology implementation clearly described?
               - Are there sufficient details for reproducibility?
               - Have appropriate validation techniques been used?
               - Are there any gaps in the experimental design?

            3. Technical Rigor:
               - Is the statistical analysis appropriate and correct?
               - Have proper controls and validation steps been included?
               - Are experimental procedures properly documented?
               - Is there evidence of methodological consistency?

            Analysis Steps:
            1) Thought: Examine methodology components and their appropriateness
            2) Action: Evaluate implementation details and technical soundness
            3) Observation: Identify any methodological gaps or weaknesses
            4) Thought: Assess severity of identified issues
            5) Action: Determine if methodological issues are severe enough to affect publishability

            Remember: Papers with inappropriate methodologies or insufficient justification 
            should be flagged as potentially non-publishable.

            Previous patterns: {patterns}

            Format your analysis using ReACT steps, then provide:
            STRENGTHS: [list with specific evidence]
            WEAKNESSES: [list critical issues found]
            SCORE: [0-1, reflecting severity of issues]
            REASONING: [detailed explanation focusing on critical problems]"""),
            ("human", "{content}")
        ])
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize methodology analyses with focus on critical issues:

            Analysis Framework:
            1. Methodology Appropriateness:
               - Evaluate overall suitability of methods
               - Assess justification quality
               - Check context adaptation

            2. Implementation Quality:
               - Review completeness of methodology
               - Assess reproducibility
               - Evaluate validation approaches

            3. Technical Soundness:
               - Check statistical rigor
               - Verify experimental design
               - Assess methodology consistency

            Previous patterns: {patterns}

            Synthesize findings focusing on:
            - Critical methodological flaws
            - Implementation gaps
            - Technical irregularities
            - Overall methodology quality

            Format final assessment with:
            STRENGTHS: [consolidated strengths]
            WEAKNESSES: [critical issues identified]
            SCORE: [0-1]
            REASONING: [comprehensive explanation]"""),
            ("human", "{analyses}")
        ])
        
        
        self.chunk_chain = self.chunk_prompt | self.llm | StrOutputParser()
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()

class ResearchObjectiveAgent(BaseAnalysisAgent):
    def __init__(self):
        super().__init__("research_objective")

        # self.llm = ChatGroq(
        #     temperature=0,
        #     api_key="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",
        #     model_name="mixtral-8x7b-32768"
        # )
        
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research objectives expert evaluating papers for critical issues in problem formulation and goals.

            Critical Issues to Identify:
            1. Problem Definition:
               - Is the research problem clearly defined?
               - Is the problem significant and worthy of study?
               - Are research gaps properly identified?
               - Are objectives aligned with stated problems?

            2. Goal Clarity:
               - Are research objectives clearly articulated?
               - Is there logical coherence in goal structure?
               - Are objectives specific and measurable?
               - Is the scope appropriate and well-justified?

            3. Research Value:
               - Is the potential impact clearly demonstrated?
               - Are objectives novel and contributing to the field?
               - Is there clear justification for the research?
               - Are claims about significance realistic?

            Analysis Steps:
            1) Thought: Analyze clarity and significance of objectives
            2) Action: Evaluate alignment with research gaps
            3) Observation: Check logical coherence of goals
            4) Thought: Assess potential impact claims
            5) Action: Identify any critical issues in problem formulation

            Remember: Papers with unclear objectives, poor problem definition, 
            or unrealistic claims should be carefully evaluated.

            Previous patterns: {patterns}

            Format your analysis using ReACT steps, then provide:
            STRENGTHS: [list with specific evidence]
            WEAKNESSES: [list critical issues found]
            SCORE: [0-1, reflecting severity of issues]
            REASONING: [detailed explanation focusing on critical problems]"""),
            ("human", "{content}")
        ])
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize research objective analyses with focus on critical issues:

            Analysis Framework:
            1. Problem Formulation:
               - Evaluate problem clarity
               - Assess significance
               - Check gap identification

            2. Objective Quality:
               - Review goal clarity
               - Assess logical structure
               - Evaluate scope appropriateness

            3. Impact Assessment:
               - Verify impact claims
               - Check novelty assertions
               - Evaluate contribution potential

            Previous patterns: {patterns}

            Synthesize findings focusing on:
            - Critical problems in objective setting
            - Goal clarity issues
            - Unrealistic claims
            - Overall objective quality

            Format final assessment with:
            STRENGTHS: [consolidated strengths]
            WEAKNESSES: [critical issues identified]
            SCORE: [0-1]
            REASONING: [comprehensive explanation]"""),
            ("human", "{analyses}")
        ])
        
        self.chunk_chain = self.chunk_prompt | self.llm | StrOutputParser()
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()


class ResultsAgent(BaseAnalysisAgent):
    def __init__(self):
        super().__init__("results")

        # self.llm = ChatGroq(
        #     temperature=0,
        #     api_key="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",
        #     model_name="mixtral-8x7b-32768"
        # )
        
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a results analysis expert evaluating papers for critical issues in research findings and their validation.

            Critical Issues to Identify:
            1. Result Validity:
               - Are results properly validated?
               - Is there sufficient evidence for claims?
               - Are findings statistically sound?
               - Are there any unrealistic or unusual results?

            2. Result Interpretation:
               - Are interpretations logical and well-supported?
               - Is there proper analysis of findings?
               - Are limitations properly acknowledged?
               - Are conclusions justified by the data?

            3. Result Presentation:
               - Are results clearly presented?
               - Is there appropriate use of statistics?
               - Are visualizations accurate and appropriate?
               - Is there sufficient detail for verification?

            Analysis Steps:
            1) Thought: Examine result validity and evidence
            2) Action: Evaluate interpretation and conclusions
            3) Observation: Check for unrealistic findings
            4) Thought: Assess support for claims
            5) Action: Identify any critical issues in results

            Remember: Papers with unsubstantiated claims, unrealistic results, 
            or improper validation should be flagged as potentially non-publishable.

            Previous patterns: {patterns}

            Format your analysis using ReACT steps, then provide:
            STRENGTHS: [list with specific evidence]
            WEAKNESSES: [list critical issues found]
            SCORE: [0-1, reflecting severity of issues]
            REASONING: [detailed explanation focusing on critical problems]"""),
            ("human", "{content}")
        ])
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize results analyses with focus on critical issues:

            Analysis Framework:
            1. Validity Assessment:
               - Review result validation
               - Check evidence quality
               - Evaluate statistical soundness

            2. Interpretation Quality:
               - Assess logical analysis
               - Check conclusion validity
               - Review limitation handling

            3. Presentation Clarity:
               - Evaluate result clarity
               - Check statistical presentation
               - Assess detail sufficiency

            Previous patterns: {patterns}

            Synthesize findings focusing on:
            - Critical issues in results
            - Validation problems
            - Interpretation flaws
            - Overall result quality

            Format final assessment with:
            STRENGTHS: [consolidated strengths]
            WEAKNESSES: [critical issues identified]
            SCORE: [0-1]
            REASONING: [comprehensive explanation]"""),
            ("human", "{analyses}")
        ])
        
        self.chunk_chain = self.chunk_prompt | self.llm | StrOutputParser()
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()


class NoveltyClaimsAgent(BaseAnalysisAgent):
    def __init__(self):
        super().__init__("novelty_claims")

        # self.llm = ChatGroq(
        #     temperature=0,
        #     api_key="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",
        #     model_name="mixtral-8x7b-32768"
        # )
        
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a novelty assessment expert evaluating papers for critical issues in innovation claims and contributions.

            Critical Issues to Identify:
            1. Claim Substantiation:
               - Are novelty claims properly supported?
               - Is there evidence for innovative aspects?
               - Are comparisons with existing work accurate?
               - Are contribution claims realistic?

            2. Innovation Assessment:
               - Is the innovation clearly articulated?
               - Are advantages properly demonstrated?
               - Is the advancement significant?
               - Are limitations acknowledged?

            3. Field Impact:
               - Is the contribution to the field clear?
               - Are impact claims realistic?
               - Is there proper positioning in literature?
               - Is the innovation scope appropriate?

            Analysis Steps:
            1) Thought: Examine novelty claims and evidence
            2) Action: Evaluate innovation significance
            3) Observation: Check claim substantiation
            4) Thought: Assess impact realism
            5) Action: Identify any critical issues in claims

            Remember: Papers with unsubstantiated novelty claims, 
            exaggerated contributions, or insufficient evidence 
            should be flagged as potentially non-publishable.

            Previous patterns: {patterns}

            Format your analysis using ReACT steps, then provide:
            STRENGTHS: [list with specific evidence]
            WEAKNESSES: [list critical issues found]
            SCORE: [0-1, reflecting severity of issues]
            REASONING: [detailed explanation focusing on critical problems]"""),
            ("human", "{content}")
        ])
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize novelty analyses with focus on critical issues:

            Analysis Framework:
            1. Claim Quality:
               - Review claim support
               - Check evidence quality
               - Assess comparison accuracy

            2. Innovation Value:
               - Evaluate innovation clarity
               - Check advantage demonstration
               - Assess significance

            3. Impact Evaluation:
               - Review field contribution
               - Check claim realism
               - Assess positioning quality

            Previous patterns: {patterns}

            Synthesize findings focusing on:
            - Critical issues in claims
            - Evidence problems
            - Impact exaggerations
            - Overall novelty quality

            Format final assessment with:
            STRENGTHS: [consolidated strengths]
            WEAKNESSES: [critical issues identified]
            SCORE: [0-1]
            REASONING: [comprehensive explanation]"""),
            ("human", "{analyses}")
        ])
        
        self.chunk_chain = self.chunk_prompt | self.llm | StrOutputParser()
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()

class DecisionMakingAgent:
    """Final decision agent using ReACT reasoning based on strict publication criteria"""
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            api_key="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",
            model_name="llama-3.3-70b-versatile"
        )
        
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the final decision maker evaluating paper publishability.
            Your role is to strictly evaluate papers based on critical criteria that determine
            publication suitability. A paper must be classified as either "Publishable" or 
            "Non-Publishable" based on the following strict criteria:

            Critical Issues that Make a Paper Non-Publishable:
            1. Methodology Problems:
               - Inappropriate methodologies for the problem
               - Lack of adequate justification for chosen methods
               - Poor adaptation of methods to the research context
               
            2. Argument Quality:
               - Unclear or disorganized arguments
               - Lack of logical coherence
               - Poor connection between sections
               
            3. Claims and Evidence:
               - Unsubstantiated claims
               - Unusually high or unrealistic results without proper validation
               - Insufficient evidence for major conclusions
               
            Evaluation Process:
            1) Thought: Carefully review each section's analysis for critical issues
            2) Action: Identify any methodology, argument, or evidence problems
            3) Observation: Assess how severe any identified issues are
            4) Thought: Determine if issues are severe enough to make paper non-publishable
            5) Action: Make final decision with clear reasoning
            
            Remember:
            - A single critical issue in methodology, argument structure, or evidence 
              validation is sufficient to classify a paper as non-publishable
            - All major claims must be properly supported with evidence
            - Methods must be appropriate and well-justified
            - Arguments must be clear and logically coherent
            
            Previous similar papers showed these patterns:
            {patterns}
            
            Format your response as:
            Thought: [Initial review of sections]
            Action: [Analysis steps]
            Observation: [Key findings]
            Thought: [Decision consideration]
            Action: [Final decision]
            
            Final output must be in format:
            DECISION: [publishable/non-publishable]
            CONFIDENCE: [0-1 score]
            REASONING: [detailed reasoning with specific issues/good parts identified]"""),
            ("human", "Section analyses: {analyses}")
        ])
        
        self.chain = self.decision_prompt | self.llm | StrOutputParser()
    
    def decide(self, analyses: Dict[str, SectionAnalysis], 
               patterns: List[Dict[str, Any]]) -> Tuple[str, str, float]:
        # Format analyses for prompt
        analyses_text = self._format_analyses(analyses)
        patterns_text = self._format_patterns(patterns)
        
        result = self.chain.invoke({
            "analyses": analyses_text,
            "patterns": patterns_text
        })
        
        # Parse decision
        decision = self._extract_value(result, "DECISION")
        confidence = self._extract_value(result, "CONFIDENCE")
        reasoning = self._extract_value(result, "REASONING")
        
        return decision, reasoning, confidence
    
    def _format_analyses(self, analyses: Dict[str, SectionAnalysis]) -> str:
        formatted = []
        for section_type, analysis in analyses.items():
            formatted.append(
                f"{section_type.upper()}:\n"
                f"Score: {analysis.contribution_score}\n"
                f"Strengths: {', '.join(analysis.strengths)}\n"
                f"Weaknesses: {', '.join(analysis.weaknesses)}\n"
                f"Reasoning: {analysis.reasoning}\n"
            )
        return "\n".join(formatted)
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        if not patterns:
            return "No previous patterns available."
        
        formatted = []
        for p in patterns[-3:]:
            formatted.append(
                f"- Decision: {p['decision']}, "
                f"Confidence: {p['confidence']}, "
                f"Key reasoning: {p['reasoning'][:100]}..."
            )
        return "\n".join(formatted)
    
    def _extract_value(self, text: str, marker: str) -> str:
        try:
            return text.split(f"{marker}:")[1].split("\n")[0].strip()
        except:
            return ""

class MemoryManager:
    """Enhanced memory manager with pattern learning"""
    
    def __init__(self):
        self.section_patterns = {}
        self.decision_patterns = []
        self.max_patterns = 20
    
    def update_memory(self, paper_state: PaperState):
        # Update section patterns
        for section_type, analysis in paper_state.section_analyses.items():
            if section_type not in self.section_patterns:
                self.section_patterns[section_type] = []
            
            pattern = {
                "content": analysis.content,
                "contribution_score": analysis.contribution_score,
                "strengths": analysis.strengths,
                "weaknesses": analysis.weaknesses,
                "reasoning": analysis.reasoning,
                "decision": paper_state.final_decision,
                "category": paper_state.features.get("category", "")
            }
            
            self.section_patterns[section_type].append(pattern)
            self.section_patterns[section_type] = self.section_patterns[section_type][-self.max_patterns:]
        
        # Update decision patterns
        decision_pattern = {
            "features": paper_state.features,
            "section_scores": {k: v.contribution_score for k, v in paper_state.section_analyses.items()},
            "decision": paper_state.final_decision,
            "confidence": paper_state.confidence,
            "reasoning": paper_state.reasoning
        }
        self.decision_patterns.append(decision_pattern)
        self.decision_patterns = self.decision_patterns[-self.max_patterns:]
    
    def get_relevant_patterns(self, 
                            section_type: str = None, 
                            category: str = None) -> List[Dict[str, Any]]:
        if section_type:
            patterns = self.section_patterns.get(section_type, [])
            if category:
                patterns = [p for p in patterns if p.get("category") == category]
            return patterns
        return self.decision_patterns

def create_evaluation_workflow(memory_manager: MemoryManager):
    """Creates the evaluation workflow graph"""
    
    # Initialize agents
    section_agents = {
        "methodology": MethodologyAgent(),
        "research_objective": ResearchObjectiveAgent(),
        "results": ResultsAgent(),
        "novelty_claims": NoveltyClaimsAgent()
    }
    
    decision_agent = DecisionMakingAgent()
    
    def process_paper(state: SystemState) -> SystemState:
        if not state.papers_queue:
            return state
        
        current_paper = state.papers_queue[0]
        
        # Analyze each section with specialized agents
        for section_type, agent in section_agents.items():
            content = current_paper.features.get(section_type, "")
            patterns = memory_manager.get_relevant_patterns(
                section_type, 
                current_paper.features.get("category")
            )
            analysis = agent.analyze(content, patterns)
            current_paper.section_analyses[section_type] = analysis
        
        # Make final decision
        patterns = memory_manager.get_relevant_patterns(
            category=current_paper.features.get("category")
        )
        decision, reasoning, confidence = decision_agent.decide(
            current_paper.section_analyses,
            patterns
        )
        
        current_paper.final_decision = decision
        current_paper.reasoning = reasoning
        current_paper.confidence = confidence
        
        # Update state and memory
        state.processed_papers.append(current_paper)
        state.papers_queue = state.papers_queue[1:]
        memory_manager.update_memory(current_paper)
        
        return state
    
    # Create graph
    workflow = StateGraph(SystemState)
    workflow.add_node("process_paper", process_paper)
    workflow.set_entry_point("process_paper")
    
    def should_continue(state: SystemState) -> str:
        return "process_paper" if state.papers_queue else END
    
    workflow.add_conditional_edges(
        "process_paper",
        should_continue
    )
    
    return workflow.compile()

def evaluate_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to evaluate papers"""
    
    # Initialize components
    memory_manager = MemoryManager()
    workflow = create_evaluation_workflow(memory_manager)
    
    # Prepare initial state
    papers_queue = []
    for idx, row in df.iterrows():
        features = {
            "methodology": str(row.get("Methodology", "")),
            "research_objective": str(row.get("Research Objective", "")),
            "results": str(row.get("Results", "")),
            "novelty_claims": str(row.get("Novelty Claims", "")),
            "category": str(row.get("Category of Research", ""))
        }
        papers_queue.append(PaperState(paper_id=str(idx), features=features))
    
    initial_state = SystemState(
        papers_queue=papers_queue,
        processed_papers=[],
    )
    
    # Run evaluation
    final_state = workflow.invoke(initial_state)
    global final_state_out
    final_state_out = final_state
    
    # Prepare results DataFrame
    results = []
    for paper in final_state['processed_papers']:
        result = {
            'paper_id': paper.paper_id,
            'decision': paper.final_decision,
            'confidence': paper.confidence,
            'reasoning': paper.reasoning,
            'section_scores': {k: v.contribution_score for k, v in paper.section_analyses.items()}
        }
        results.append(result)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    
    # Evaluate papers
    df = pd.read_csv('paper_df.csv')
    results_df_test = evaluate_papers(df)
    print(results_df_test)