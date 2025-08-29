"""
Comprehensive Project Glossary and Knowledge Base for LUKi AI

This module contains the complete glossary of terms, concepts, and ecosystem knowledge
for the LUKi AI agent to have full understanding of ReMeLife platform.
"""

from typing import Dict, List, Any


class ProjectKnowledgeBase:
    """Comprehensive knowledge base for LUKi AI containing all project context."""
    
    @staticmethod
    def get_glossary() -> Dict[str, str]:
        """Get comprehensive glossary of all ReMeLife/LUKi terms."""
        return {
            # Core Identity Terms
            "LUKi": "Primary AI companion in ReMeLife ecosystem. Memory-augmented assistant specializing in person-centered care for dementia, cognitive decline, and learning disabilities. Powered by LLaMA 3.3 70B.",
            
            "ReMeLife": "Web3-based care ecosystem revolutionizing person-centered care through tokenized rewards and AI companionship. Includes ReMeCare app, Community platform, Market, Forum, and Wallet.",
            
            "ELR": "Electronic Life Records - Proprietary data structure capturing life stories, preferences, engagement patterns, family context, health journey, and daily activities for personalized AI interactions.",
            
            # Token Economy Terms
            "CAP": "Care Action Points - Non-transferable points earned for performing care actions in RemindMeCare app. Unlimited supply, converted to REME tokens at 60:1 ratio. Foundation of care reward system.",
            
            "REME": "Primary utility token of ReMeLife ecosystem. Fixed supply of 2 billion tokens. Used for marketplace purchases, staking, governance. Earned through CAP conversion and care activities.",
            
            "LUKi Token": "AI participation rewards for users contributing to federated learning system. Future implementation for decentralized AI training rewards.",
            
            # Platform Components
            "ReMeCare": "Also known as RemindMeCare - B2B care provider application offering 350+ digital activities for dementia and learning disability support. Core data source for ELR.",
            
            "RemindMeCare": "Original name of ReMeCare app. Professional care platform with activity library, engagement tracking, and family connection features.",
            
            "ReMe Market": "Marketplace within ReMeLife ecosystem where users spend REME tokens on care-related products and services with additional token rewards.",
            
            "ReMe Forum": "Community support platform for carers, families, and care professionals to share experiences and resources.",
            
            "ReMe Wallet": "Token management system for CAPs and REME tokens with referral rewards and transaction history.",
            
            # Care Concepts
            "Person-Centered Care": "Care approach focusing on individual preferences, life history, and maintaining dignity regardless of cognitive capacity.",
            
            "Care Circle": "Network of family members, professional carers, and support workers involved in an individual's care.",
            
            "My Story": "Feature in RemindMeCare for storing personal life history, photos, videos, and memories for therapeutic reminiscence.",
            
            "World Days Calendar": "Themed activity suggestions based on international awareness days integrated into RemindMeCare.",
            
            "Activity Engagement": "Measurable interactions with RemindMeCare activities used to build ELR patterns and earn CAPs.",
            
            # Technical Terms
            "Vector Embeddings": "Semantic representations of ELR data stored in ChromaDB for intelligent retrieval and personalization.",
            
            "RAG": "Retrieval-Augmented Generation - Technique combining vector search with LLM generation for accurate, contextual responses.",
            
            "Federated Learning": "Future phase allowing distributed AI training while preserving privacy. Users earn LUKi tokens for participation.",
            
            "ChromaDB": "Vector database storing ELR embeddings for semantic memory search and retrieval.",
            
            "LangChain": "Framework orchestrating LUKi's conversation flow, tool routing, and memory integration.",
            
            # Care Terminology
            "Dementia": "Progressive neurological condition affecting memory, thinking, and behavior. LUKi provides specialized support through validation therapy and reminiscence.",
            
            "Learning Disabilities": "Lifelong conditions affecting learning and daily living skills. LUKi adapts communication for accessibility and empowerment.",
            
            "Validation Therapy": "Therapeutic approach accepting and validating emotions rather than correcting false memories. Core to LUKi's dementia care.",
            
            "Reminiscence Therapy": "Using life memories and familiar content to stimulate cognitive function and emotional wellbeing.",
            
            "Sundowning": "Increased confusion and agitation in dementia patients during late afternoon/evening. LUKi adjusts interaction style accordingly.",
            
            # Governance & Economics
            "Care Mining": "Earning tokens through care activities, similar to crypto mining but for social good.",
            
            "Care Staking": "Locking REME tokens to earn rewards and participate in platform governance.",
            
            "DAO": "Decentralized Autonomous Organization - Future governance structure for ReMeLife platform decisions.",
            
            "Tokenomics": "Economic model of CAP/REME/LUKi tokens incentivizing quality care through blockchain rewards.",
            
            # Privacy & Security
            "Data Sovereignty": "Users maintain ownership and control of their ELR data with consent-based sharing.",
            
            "GDPR Compliance": "Full compliance with European data protection regulations including right to erasure.",
            
            "Consent Levels": "Granular privacy controls: Private, Family, Care Team, Medical, Research.",
            
            "Zero-Knowledge Proofs": "Future cryptographic method for verifying care actions without revealing personal data."
        }
    
    @staticmethod
    def get_platform_knowledge() -> Dict[str, Any]:
        """Get structured knowledge about the ReMeLife platform."""
        return {
            "platform_overview": {
                "name": "ReMeLife",
                "mission": "Revolutionize person-centered care through Web3 technology and AI companionship",
                "target_users": [
                    "Individuals with dementia or cognitive decline",
                    "People with learning disabilities", 
                    "Family carers and care circles",
                    "Professional care providers and organizations",
                    "Healthcare systems seeking cost-effective care solutions"
                ],
                "value_proposition": "Transform care from cost center to rewarded activity through tokenization"
            },
            
            "token_mechanics": {
                "cap_earning": {
                    "activities": ["Completing care activities", "Family engagement", "Assessment participation"],
                    "rate": "Variable based on activity complexity and engagement quality",
                    "restrictions": "Non-transferable, bound to user account"
                },
                "reme_conversion": {
                    "ratio": "60 CAP = 1 REME",
                    "vesting": "Immediate conversion available",
                    "uses": ["Marketplace purchases", "Premium features", "Staking for rewards", "Governance voting"]
                },
                "luki_rewards": {
                    "earning": "Contributing ELR data for federated learning",
                    "future_implementation": "Phase 3 - Federated Learning rollout",
                    "privacy": "Differential privacy ensures individual data protection"
                }
            },
            
            "care_activities": {
                "categories": [
                    "Music Therapy",
                    "Reminiscence Activities", 
                    "Cognitive Stimulation",
                    "Physical Movement",
                    "Creative Arts",
                    "Social Connection",
                    "Sensory Experiences",
                    "Life Skills Support"
                ],
                "total_activities": "350+ activities in RemindMeCare library",
                "personalization": "Activities recommended based on ELR patterns and preferences",
                "tracking": "Engagement metrics feed into ELR for continuous improvement"
            },
            
            "technical_architecture": {
                "ai_model": "LLaMA 3.3 70B Instruct Turbo",
                "memory_system": "ChromaDB vector store with semantic search",
                "framework": "LangChain for orchestration and tool routing",
                "database": "MySQL for RemindMeCare data, vector DB for embeddings",
                "deployment": "Phase 1 - Local/API, Phase 2 - AWS/GPU cluster",
                "privacy": "Local processing priority, API usage for testing only"
            }
        }
    
    @staticmethod
    def get_conversation_context() -> str:
        """Get essential context for conversation initialization."""
        return """
You are LUKi, the primary AI companion in the ReMeLife ecosystem. You have comprehensive knowledge of:

1. **Your Identity**: Memory-augmented assistant specializing in dementia and learning disability care
2. **ReMeLife Platform**: Complete understanding of ReMeCare app, token economy (CAP/REME), and care ecosystem
3. **Care Philosophy**: Person-centered, validation-first approach with dignity preservation
4. **Technical Capabilities**: ELR processing, vector embeddings, personalized recommendations
5. **Token Economy**: How users earn CAPs through care activities and convert to REME tokens

When responding:
- Always ground responses in your comprehensive platform knowledge
- Reference specific features (RemindMeCare activities, My Story, World Days)
- Explain token mechanics accurately (60 CAP = 1 REME, etc.)
- Maintain your warm, mischievous personality while being informative
- Use ELR data for personalization when available
"""
    
    @staticmethod
    def get_activity_categories() -> List[Dict[str, Any]]:
        """Get detailed RemindMeCare activity categories for recommendations."""
        return [
            {
                "category": "Music & Entertainment",
                "activities": ["Sing-along sessions", "Music quiz", "Dance therapy", "Instrument play"],
                "cap_value": 10,
                "suitable_for": ["dementia", "learning_disabilities", "general"]
            },
            {
                "category": "Reminiscence",
                "activities": ["Photo sharing", "Life story telling", "Memory boxes", "Then and now"],
                "cap_value": 15,
                "suitable_for": ["dementia", "general"]
            },
            {
                "category": "Cognitive Stimulation",
                "activities": ["Word games", "Number puzzles", "Memory exercises", "Trivia"],
                "cap_value": 12,
                "suitable_for": ["mild_cognitive_impairment", "learning_disabilities"]
            },
            {
                "category": "Creative Arts",
                "activities": ["Painting", "Crafts", "Poetry", "Collage making"],
                "cap_value": 10,
                "suitable_for": ["all"]
            },
            {
                "category": "Physical Movement",
                "activities": ["Chair exercises", "Walking group", "Gardening", "Ball games"],
                "cap_value": 8,
                "suitable_for": ["all"]
            },
            {
                "category": "Social Connection",
                "activities": ["Video calls", "Letter writing", "Group discussions", "Tea time"],
                "cap_value": 10,
                "suitable_for": ["all"]
            },
            {
                "category": "Life Skills",
                "activities": ["Cooking together", "Sorting tasks", "Folding", "Planning"],
                "cap_value": 12,
                "suitable_for": ["learning_disabilities", "mild_dementia"]
            },
            {
                "category": "Sensory",
                "activities": ["Aromatherapy", "Hand massage", "Texture exploration", "Sound therapy"],
                "cap_value": 8,
                "suitable_for": ["advanced_dementia", "learning_disabilities"]
            }
        ]


def get_comprehensive_glossary() -> Dict[str, str]:
    """Convenience function to get glossary."""
    kb = ProjectKnowledgeBase()
    return kb.get_glossary()


def get_platform_context() -> str:
    """Get complete platform context for LLM system prompt."""
    kb = ProjectKnowledgeBase()
    glossary = kb.get_glossary()
    platform = kb.get_platform_knowledge()
    
    # Create comprehensive context
    context = kb.get_conversation_context()
    context += "\n\n## Key Terms You Must Know:\n"
    
    # Add critical terms
    critical_terms = ["CAP", "REME", "ELR", "ReMeCare", "RemindMeCare", "LUKi Token"]
    for term in critical_terms:
        if term in glossary:
            context += f"- **{term}**: {glossary[term]}\n"
    
    return context
