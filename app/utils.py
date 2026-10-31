"""
Utility Functions
Helper functions for the QA system
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Tuple


# Performance data from training
PERFORMANCE_DATA = {
    'English': {
        'BLEU': 37.79,
        'ROUGE-1': 0.6282,
        'ROUGE-2': 0.3710,
        'ROUGE-L': 0.6272,
        'Exact Match': 0.4360,
        'F1 Score': 0.6329,
        'Avg (EM+F1)': 0.5344
    },
    'German': {
        'BLEU': 43.12,
        'ROUGE-1': 0.6646,
        'ROUGE-2': 0.4064,
        'ROUGE-L': 0.6622,
        'Exact Match': 0.4874,
        'F1 Score': 0.6580,
        'Avg (EM+F1)': 0.5727
    }
}


def calculate_confidence(answer: str, context: str) -> str:
    """
    Calculate answer confidence level
    
    Args:
        answer: Generated answer
        context: Input context
        
    Returns:
        Confidence level: "High", "Medium", or "Low"
    """
    if len(answer.split()) < 2:
        return "Low"
    elif answer.lower() in context.lower():
        return "High"
    else:
        return "Medium"


def format_answer(answer: str, language: str, confidence: str) -> str:
    """
    Format answer with metadata
    
    Args:
        answer: Generated answer
        language: Language used
        confidence: Confidence level
        
    Returns:
        Formatted string with answer details
    """
    return f"""
### 📊 Response Details
- **Language**: {language}
- **Answer Length**: {len(answer.split())} words
- **Confidence**: {confidence}
- **Model**: mBART-large-50 + LoRA
    """


def create_performance_chart() -> go.Figure:
    """
    Create interactive performance comparison chart
    
    Returns:
        Plotly figure object
    """
    metrics = ['BLEU', 'ROUGE-L', 'Exact Match', 'F1 Score']
    
    english_scores = [
        PERFORMANCE_DATA['English']['BLEU'] / 100,
        PERFORMANCE_DATA['English']['ROUGE-L'],
        PERFORMANCE_DATA['English']['Exact Match'],
        PERFORMANCE_DATA['English']['F1 Score']
    ]
    
    german_scores = [
        PERFORMANCE_DATA['German']['BLEU'] / 100,
        PERFORMANCE_DATA['German']['ROUGE-L'],
        PERFORMANCE_DATA['German']['Exact Match'],
        PERFORMANCE_DATA['German']['F1 Score']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='English', x=metrics, y=english_scores, marker_color='#3498db'),
        go.Bar(name='German', x=metrics, y=german_scores, marker_color='#e74c3c')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison: English vs German',
        xaxis_title='Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        barmode='group',
        template='plotly_white',
        height=400,
        font=dict(size=12)
    )
    
    return fig


def create_metrics_table() -> pd.DataFrame:
    """
    Create detailed metrics table
    
    Returns:
        Pandas DataFrame with metrics
    """
    df = pd.DataFrame(PERFORMANCE_DATA).T
    df = df.round(4)
    return df


def get_example(example_type: str, language: str) -> Tuple[str, str]:
    """
    Get example question and context
    
    Args:
        example_type: Type of example ("General Knowledge", "Historical", "Scientific")
        language: "English" or "German"
        
    Returns:
        Tuple of (question, context)
    """
    examples = {
        "English": {
            "General Knowledge": (
                "What is the capital of France?",
                "Paris is the capital and most populous city of France. It has an area of 105 square kilometres and a population of 2,165,423 residents."
            ),
            "Historical": (
                "When was the Eiffel Tower built?",
                "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair."
            ),
            "Scientific": (
                "What is the largest planet in our solar system?",
                "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than two and a half times that of all the other planets combined."
            )
        },
        "German": {
            "General Knowledge": (
                "Was ist die Hauptstadt von Deutschland?",
                "Berlin ist die Hauptstadt und größte Stadt Deutschlands mit etwa 3,7 Millionen Einwohnern."
            ),
            "Historical": (
                "Wann wurde der Berliner Fernsehturm gebaut?",
                "Der Berliner Fernsehturm wurde zwischen 1965 und 1969 erbaut und ist eines der bekanntesten Wahrzeichen Berlins."
            ),
            "Scientific": (
                "Was ist der größte Planet in unserem Sonnensystem?",
                "Jupiter ist der größte Planet in unserem Sonnensystem. Er ist ein Gasriese mit einer Masse, die mehr als zweieinhalb Mal so groß ist wie die aller anderen Planeten zusammen."
            )
        }
    }
    
    return examples[language][example_type]