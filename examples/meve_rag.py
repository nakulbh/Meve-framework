import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from meve.core.engine import MeVeEngine

# Import MeVe components
from meve.core.models import ContextChunk, MeVeConfig
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger


class SimpleRAGSystem:
    def __init__(self, collection_name: str = "hotpotqa_sentences"):
        self.collection_name = collection_name
        self.chunks: List[ContextChunk] = []
        self.engine: Optional[MeVeEngine] = None
        self.vector_db_client: Optional[VectorDBClient] = None
        self.openai_client: Optional[OpenAI] = None

        # Create unique instance ID for logging
        self.instance_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Setup logging directories
        self.log_dir = Path("logs/meve_contexts")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_log_dir = self.log_dir / self.instance_id
        self.session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger for this instance
        self.logger = get_logger(__name__)
        
        # Log initialization
        self._log_session_start()

        # Load chunks from ChromaDB and initialize engine
        self.load_from_chromadb()
        self.initialize_engine()
        self.initialize_openai()

    def _log_session_start(self) -> None:
        """Log session initialization details."""
        self.logger.info(f"🚀 Starting MeVe RAG Session: {self.instance_id}")
        
        session_info = {
            "session_id": self.instance_id,
            "timestamp": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "log_directory": str(self.session_log_dir),
            "status": "initialized"
        }
        
        # Save session metadata
        session_file = self.session_log_dir / "session_info.json"
        with open(session_file, "w") as f:
            json.dump(session_info, f, indent=2)
        
        self.logger.info(f"📋 Session info saved to: {session_file}")

    def load_from_chromadb(self) -> None:
        """Load chunks from ChromaDB collection."""
        self.logger.info(f"📥 Loading data from ChromaDB collection: {self.collection_name}")

        try:
            # Get ChromaDB Cloud credentials from environment variables
            chroma_config = {
                "api_key": os.getenv("CHROMA_API_KEY"),
                "tenant": os.getenv("CHROMA_TENANT"),
                "database": os.getenv("CHROMA_DATABASE"),
            }

            # Validate that all required env vars are set
            if not all(chroma_config.values()):
                missing = [k for k, v in chroma_config.items() if not v]
                raise ValueError(
                    f"Missing ChromaDB Cloud credentials: {', '.join(missing)}. "
                    f"Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE environment variables in .env file."
                )

            self.vector_db_client = VectorDBClient(
                chunks=None,
                is_persistent=True,
                collection_name=self.collection_name,
                load_existing=True,
                use_cloud=True,
                cloud_config=chroma_config,
                embedding_model="all-MiniLM-L6-v2",
            )

            # Get the chunks from the client
            self.chunks = self.vector_db_client.chunks
            self.logger.info(f"✅ Loaded {len(self.chunks)} chunks from ChromaDB")
            
            # Log chunk statistics
            self._log_chunk_stats()

        except Exception as e:
            self.logger.error(f"❌ Failed to load from ChromaDB: {e}")
            self.logger.warn("💡 Make sure the ChromaDB collection exists")
            self.chunks = []
            self.vector_db_client = None

    def _log_chunk_stats(self) -> None:
        """Log statistics about loaded chunks."""
        if not self.chunks:
            return
        
        chunk_stats = {
            "total_chunks": len(self.chunks),
            "timestamp": datetime.now().isoformat(),
            "chunk_samples": [
                {
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,  # Full content, no truncation
                    "content_length": len(chunk.content),
                    "has_embedding": chunk.embedding is not None,
                }
                for chunk in self.chunks[:5]  # First 5 samples
            ]
        }
        
        stats_file = self.session_log_dir / "chunk_stats.json"
        with open(stats_file, "w") as f:
            json.dump(chunk_stats, f, indent=2)
        
        self.logger.info(f"📊 Chunk statistics saved to: {stats_file}")

    def initialize_engine(self) -> None:
        """Initialize the MeVe engine with ChromaDB vector client."""
        if not self.chunks or not self.vector_db_client:
            self.logger.error("❌ No chunks or vector client available for engine initialization")
            return

        self.logger.info("⚙️  Initializing MeVe engine...")

        # Create default config (you can customize these parameters)
        config = MeVeConfig(
            k_init=100,  # Initial retrieval count
            tau_relevance=0.2,  # Relevance threshold (lowered from 0.3 for more results)
            n_min=5,  # Minimum verified docs (lowered from 50 for earlier completion)
            theta_redundancy=0.8,  # Redundancy threshold
            lambda_mmr=0.5,  # MMR lambda
            t_max=1000,  # Token budget (increased from 900 for more context)
            embedding_model="all-MiniLM-L6-v2",  # OpenAI Text Embedding 3 Small
        )

        # Log configuration
        self._log_config(config)

        try:
            # Convert chunks list to dict for BM25 index
            bm25_index = {chunk.doc_id: chunk for chunk in self.chunks}

            # Initialize with VectorDBClient for phase 1 and dict for BM25
            self.engine = MeVeEngine(
                config=config,
                vector_db_client=self.vector_db_client,  # Use ChromaDB client
                bm25_index=bm25_index,  # Use dict for BM25
            )
            self.logger.info("✅ MeVe engine initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MeVe engine: {e}", error=str(e))
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.engine = None

    def _log_config(self, config: MeVeConfig) -> None:
        """Log MeVe configuration."""
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "k_init": config.k_init,
            "tau_relevance": config.tau_relevance,
            "n_min": config.n_min,
            "theta_redundancy": config.theta_redundancy,
            "lambda_mmr": config.lambda_mmr,
            "t_max": config.t_max,
            "embedding_model": config.embedding_model,
        }
        
        config_file = self.session_log_dir / "meve_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"⚙️  MeVe config saved to: {config_file}")

    def initialize_openai(self) -> None:
        """Initialize OpenAI client for generating natural language answers."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warn("⚠️  OPENAI_API_KEY not set - answers won't be generated")
                self.openai_client = None
                return

            self.openai_client = OpenAI(api_key=api_key)
            self.logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            self.logger.warn(f"⚠️  Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    def generate_answer_with_openai(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language answer using OpenAI based on retrieved context.

        Args:
            question: The original question
            context_chunks: List of context chunks with content and metadata

        Returns:
            Natural language answer from OpenAI or "Not enough context" message
        """
        # Check if we have sufficient context
        if not self.openai_client:
            msg = "❌ OpenAI client not available."
            self.logger.error(msg)
            return msg

        try:
            # Prepare context text from chunks
            context_text = "\n\n".join(
                [f"[Source {i + 1}] {chunk['content']}" for i, chunk in enumerate(context_chunks)]
            )

            self.logger.info(f"🤖 Generating answer with OpenAI (context chunks: {len(context_chunks)})")

            # Create a strict prompt that ONLY uses provided context
            system_prompt = """You are a helpful assistant that answers questions STRICTLY based on provided context.

IMPORTANT RULES:
1. You MUST only use information explicitly stated in the provided context.
2. If the context does not contain information needed to answer the question, respond with exactly: "Not enough context to answer this question."
3. Do not make up, infer, or use any external knowledge.
4. Cite the source number when referencing information.
5. Be clear and concise in your answers."""

            user_prompt = f"""Question: {question}

Context:
{context_text}

Answer the question ONLY using the context provided above. If the context is insufficient, state: "Not enough context to answer this question." """

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini for cost-effectiveness
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic, fact-based responses
                max_tokens=1000,
            )

            answer = response.choices[0].message.content
            self.logger.info(f"✅ OpenAI answer generated ({len(answer)} chars)")

            # Check if OpenAI itself says there's not enough context
            

            return answer

        except Exception as e:
            error_msg = f"❌ Error generating answer with OpenAI: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return error_msg

    def log_context(self, question: str, context_chunks: List[Dict[str, Any]], answer: str) -> None:
        """
        Log the retrieved context and answer to a JSON file.

        Args:
            question: The original question
            context_chunks: List of context chunks retrieved from MeVe
            answer: The generated answer
        """
        try:
            # Create log entry with full content (no truncation)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "context_count": len(context_chunks),
                "context_chunks": context_chunks,  # Full chunks with complete content
                "answer": answer,  # Full answer, no truncation
            }

            # Generate filename with timestamp
            filename = self.session_log_dir / f"qa_{datetime.now().strftime('%H%M%S_%f')}.json"

            # Write to file
            with open(filename, "w") as f:
                json.dump(log_entry, f, indent=2)

            self.logger.info(f"✅ Q&A logged to: {filename}")

        except Exception as e:
            self.logger.error(f"⚠️  Failed to log context: {e}", error=str(e))

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the MeVe RAG pipeline and OpenAI.

        Args:
            question: The question to answer

        Returns:
            Dict containing answer, context, and metadata
        """
        if not self.engine:
            self.logger.error("❌ RAG system not properly initialized")
            return {
                "answer": "RAG system not properly initialized",
                "context": [],
                "metadata": {"error": "engine_not_initialized"},
            }

        try:
            overall_start = time.time()
            self.logger.info(f"❓ Processing question: '{question}'")

            # Run MeVe pipeline
            pipeline_start = time.time()
            final_answer, final_chunks = self.engine.run(question)
            pipeline_time = time.time() - pipeline_start
            self.logger.info(f"✅ MeVe pipeline completed - Retrieved {len(final_chunks)} chunks in {pipeline_time:.2f}s")
            [
  "Were Scott Derrickson and Ed Wood of the same nationality?",
  "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
  "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
  "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
  "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
  "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
  "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
  "Who is older, Annie Morton or Terry Richardson?",
  "Are Local H and For Against both from the United States?",
  "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
  "What screenwriter with credits for \"Evolution\" co-wrote a film starring Nicolas Cage and T\u00e9a Leoni?",
  "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?",
  "Are Random House Tower and 888 7th Avenue both used for real estate?",
  "The football manager who recruited David Beckham managed Manchester United during what timeframe?",
  "Brown State Fishing Lake is in a country that has a population of how many inhabitants ?",
  "The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?",
  "Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?",
  "Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?",
  "Which writer was from England, Henry Roth or Robert Erskine Childers?",
  "Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?",
  "This singer of A Rather Blustery Day also voiced what hedgehog?",
  "Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?",
  "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? ",
  "What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?",
  "Who was the writer of These Boots Are Made for Walkin' and who died in 2007?",
  "The 2011\u201312 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?",
  "Are both Dictyosperma, and Huernia described as a genus?",
  "Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?",
  "What is the name for the adventure in \"Tunnels and Trolls\", a game designed by Ken St. Andre?",
  "When was Poison's album \"Shut Up, Make Love\" released?",
  "Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?",
  "Which  French ace pilot and adventurer fly L'Oiseau Blanc",
  "Are Freakonomics and In the Realm of the Hackers both American documentaries?",
  "Which band, Letters to Cleo or Screaming Trees, had more members?",
  "Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?",
  "Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?",
  "The Livesey Hal War Memorial commemorates the fallen of which war, that had over 60 million casualties?",
  "Are both Elko Regional Airport and Gerald R. Ford International Airport located in Michigan?",
  "Ralph Hefferline was a psychology professor at a university that is located in what city?",
  "Which dog's ancestors include Gordon and Irish Setters: the Manchester Terrier or the Scotch Collie?",
  "Where is the company that Sachin Warrier worked for as a software engineer headquartered? ",
  "A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?",
  "The battle in which Giuseppe Arimondi lost his life secured what for Ethiopia?",
  "Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under which United States Vice President?",
  "A medieval fortress in Dirleton, East Lothian, Scotland borders on the south side of what coastal area?",
  "Who is the writer of this song that was inspired by words on a tombstone and was the first track on the box set Back to Mono?",
  "What type of forum did a former Soviet statesman initiate?",
  "Are Ferocactus and Silene both types of plant?",
  "Which British first-generation jet-powered medium bomber was used in the South West Pacific theatre of World War II?",
  "Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play?",
  "In 1991 Euromarch\u00e9 was bought by a chain that operated how any hypermarkets at the end of 2016?",
  "What race track in the midwest hosts a 500 mile race eavery May?",
  "In what city did the \"Prince of tenors\" star in a film based on an opera by Giacomo Puccini?",
  "Ellie Goulding worked with what other writers on her third studio album, Delirium?",
  "Which Australian city founded in 1838 contains a boarding school opened by a Prime Minister of Australia and named after a school in London of the same name.",
  "D1NZ is a series based on what oversteering technique?",
  "who is younger Keith Bostic or Jerry Glanville ?",
  "According to the 2001 census, what was the population of the city in which Kirton End is located?",
  "Are both Cypress and Ajuga genera?",
  "What distinction is held by the former NBA player who was a member of the Charlotte Hornets during their 1992-93 season and was head coach for the WNBA team Charlotte Sting?",
  "What is the name of the executive producer of the film that has a score composed by Jerry Goldsmith?",
  "Who was born earlier, Emma Bull or Virginia Woolf?",
  "What was the Roud Folk Song Index of the nursery rhyme inspiring What Are Little Girls Made Of?",
  "Scott Parkin has been a vocal critic of Exxonmobil and another corporation that has operations in how many countries ?",
  "What WB supernatrual drama series was Jawbreaker star Rose Mcgowan best known for being in?",
  "Vince Phillips held a junior welterweight title by an organization recognized by what larger Hall of Fame?",
  "What is the name of the singer who's song was released as the lead single from the album \"Confessions\", and that had popular song stuck behind for eight consecutive weeks?",
  "who is the younger brother of The episode guest stars of The Hard Easy ",
  "The 2017\u201318 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?",
  "Which of Tara Strong major voice role in animated series is an American animated television series based on the DC Comics fictional superhero team, the \"Teen Titans\"?",
  "What is the inhabitant of the city where  122nd SS-Standarte was formed in2014",
  "What color clothing do people of the Netherlands wear during Oranjegekte or to celebrate the national holiday Koningsdag? ",
  "What was the name of the 1996 loose adaptation of William Shakespeare's \"Romeo & Juliet\" written by James Gunn?",
  "Robert Suettinger was the national intelligence officer under which former Governor of Arkansas?",
  "What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?",
  "What is the middle name of the actress who plays Bobbi Bacha in Suburban Madness?",
  "Alvaro Mexia had a diplomatic mission with which tribe of indigenous people?",
  "What nationality were social anthropologists Alfred Gell and Edmund Leach?",
  "In which year was the King who made the 1925 Birthday Honours born?",
  "What is the county seat of the county where East Lempster, New Hampshire is located?",
  "The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What was the Rock singers stage name ?",
  "Rostker v. Goldberg held that the practice of what way of filling armed forces vacancies was consitutional?",
  "Handi-Snacks are a snack food product line sold by what American multinational confectionery, food, and beverage company that is based in Illinois?",
  "What was the name of a woman from the book titled \"Their Lives: The Women Targeted by the Clinton Machine \" and was also a former white house intern?",
  "When was the American lawyer, lobbyist and political consultant who was a senior member of the presidential campaign of Donald Trump born?",
  "In what year was the novel that Louren\u00e7o Mutarelli based \"Nina\" on based first published?",
  "Where are Teide National Park and Garajonay National Park located?",
  "How many copies of Roald Dahl's variation on a popular anecdote sold?",
  "What occupation do Chris Menges and Aram Avakian share?",
  "Andrew Jaspan was the co-founder of what not-for-profit media outlet?",
  "Which American film director hosted the 18th Independent Spirit Awards in 2002?",
  "Where does the hotel and casino located in which Bill Cosby's third album was recorded?",
  "Do the drinks Gibson and Zurracapote both contain gin?",
  "In what month is the annual documentary film festival, that is presented by the fortnightly published British journal of literary essays, held? ",
  "Tysons Galleria is located in what county?",
  "Bordan Tkachuk was the CEO of a company that provides what sort of products?",
  "Which filmmaker was known for animation, Lev Yilmaz or Pamela B. Green?",
  "In which city is the ambassador of the Rabat-Sal\u00e9-K\u00e9nitra administrative region to China based?",
  "Are Yingkou and Fuding the same level of city?"
]
            # Log pipeline details
            self._log_pipeline_details(question, final_chunks, final_answer)

            # Prepare context for OpenAI
            context_list = [
                {
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "title": "Unknown",  # Extract from content if available
                    "source": "unknown",
                    "relevance_score": getattr(chunk, 'relevance_score', None),
                    "token_count": getattr(chunk, 'token_count', None),
                }
                for chunk in final_chunks
            ]

            # Generate natural language answer using OpenAI
            self.logger.info("🤖 Generating natural language answer with OpenAI...")
            openai_start = time.time()
            if context_list and self.openai_client:
                openai_answer = self.generate_answer_with_openai(question, context_list)
            elif not context_list:
                openai_answer = "❌ Not enough context available to answer this question."
                self.logger.warn("⚠️  No context retrieved from MeVe pipeline")
            else:
                openai_answer = "❌ OpenAI not configured."
                self.logger.warn("⚠️  OpenAI client not available")
            openai_time = time.time() - openai_start

            # Log the context and answer
            self.log_context(question, context_list, openai_answer)

            overall_time = time.time() - overall_start
            
            result = {
                "answer": openai_answer,
                "context": context_list,
                "metadata": {
                    "total_chunks_in_db": len(self.chunks),
                    "retrieved_chunks": len(final_chunks),
                    "used_context_chunks": len(context_list),
                    "config": {
                        "k_init": self.engine.config.k_init,
                        "tau_relevance": self.engine.config.tau_relevance,
                        "t_max": self.engine.config.t_max,
                    },
                    "timings": {
                        "pipeline_seconds": round(pipeline_time, 3),
                        "openai_seconds": round(openai_time, 3),
                        "total_seconds": round(overall_time, 3),
                    },
                    "meve_raw_answer": final_answer,
                },
            }
            
            self.logger.info(f"✅ Question processed successfully in {overall_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing question: {e}", error=str(e))
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context": [],
                "metadata": {"error": str(e)},
            }

    def _log_pipeline_details(self, question: str, chunks: List[ContextChunk], raw_answer: str) -> None:
        """Log detailed pipeline execution information."""
        pipeline_log = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "raw_answer": raw_answer,  # Full raw answer, no truncation
            "chunks_retrieved": len(chunks),
            "chunk_details": [
                {
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,  # Full content, no truncation
                    "content_length": len(chunk.content),
                    "relevance_score": getattr(chunk, 'relevance_score', None),
                    "token_count": getattr(chunk, 'token_count', None),
                }
                for chunk in chunks
            ]
        }
        
        pipeline_file = self.session_log_dir / f"pipeline_{datetime.now().strftime('%H%M%S_%f')}.json"
        with open(pipeline_file, "w") as f:
            json.dump(pipeline_log, f, indent=2)
        
        self.logger.info(f"📋 Pipeline details saved to: {pipeline_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "collection_name": self.collection_name,
            "total_chunks": len(self.chunks),
            "engine_initialized": self.engine is not None,
            "vector_db_initialized": self.vector_db_client is not None,
            "openai_initialized": self.openai_client is not None,
            "session_id": self.instance_id,
            "log_directory": str(self.session_log_dir),
        }
        
        self.logger.info(f"📊 System stats: {stats}")
        return stats


def main(questions_to_process: List[str] = None):
    """Batch Q&A CLI for processing questions.
    
    Args:
        questions_to_process: List of questions to process. If None, loads from all_questions.json
    """
    print("=" * 70)
    print("🤖 MeVe RAG Question-Answering System (Batch Mode)")
    print("=" * 70)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Show statistics
    stats = rag.get_stats()
    print(f"\n📊 System Status:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Engine: {'✅ Ready' if stats['engine_initialized'] else '❌ Failed'}")
    print(f"   Logs: {stats['log_directory']}")

    if not stats["engine_initialized"]:
        print("\n❌ RAG system failed to initialize.")
        print("⚠️  Please ensure:")
        print("   • 'hotpotqa_contexts' collection exists in ChromaDB")
        print("   • Environment variables are set:")
        print("     - CHROMA_API_KEY")
        print("     - CHROMA_TENANT")
        print("     - CHROMA_DATABASE")
        return

    if not stats["openai_initialized"]:
        print("\n⚠️  OpenAI integration not available.")
        print("   Set OPENAI_API_KEY environment variable for natural language answers.")

    # Load questions from file if not provided
    if questions_to_process is None:
        try:
            with open("all_questions.json", "r") as f:
                questions_to_process = json.load(f)
            print(f"\n✅ Loaded {len(questions_to_process)} questions from all_questions.json")
        except FileNotFoundError:
            print("\n❌ Could not find all_questions.json")
            return
    else:
        print(f"\n✅ Processing {len(questions_to_process)} selected questions")

    print("\n" + "=" * 70)
    print(f"🔄 Processing {len(questions_to_process)} questions with 1 second delay...")
    print("=" * 70 + "\n")

    # Collect statistics for report
    report_data = {
        "session_id": stats['session_id'],
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions_to_process),
        "questions_processed": 0,
        "successful": 0,
        "failed": 0,
        "total_context_chunks_retrieved": 0,
        "total_context_chunks_used": 0,
        "total_time_seconds": 0.0,
        "avg_context_chunks": 0.0,
        "avg_time_per_question": 0.0,
        "start_time": datetime.now(),
        "question_results": []
    }

    # Process all questions
    for question_num, question in enumerate(questions_to_process, 1):
        try:
            print(f"[{question_num}/{len(questions_to_process)}] ❓ Question: {question[:80]}{'...' if len(question) > 80 else ''}")

            # Get answer
            result = rag.answer_question(question)

            # Display answer (truncated for readability)
            answer_preview = result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"]
            print(f"         ✅ Answer: {answer_preview}")

            # Show retrieved context count and timing
            context_retrieved = result["metadata"]["retrieved_chunks"]
            context_used = result["metadata"]["used_context_chunks"]
            timings = result["metadata"]["timings"]
            
            print(f"         📚 Context: {context_used} used / {context_retrieved} retrieved")
            print(f"         ⏱️  Time: Pipeline {timings['pipeline_seconds']:.2f}s + OpenAI {timings['openai_seconds']:.2f}s = {timings['total_seconds']:.2f}s")
            print()

            # Collect data for report
            report_data["questions_processed"] += 1
            report_data["successful"] += 1
            report_data["total_context_chunks_retrieved"] += context_retrieved
            report_data["total_context_chunks_used"] += context_used
            report_data["total_time_seconds"] += timings['total_seconds']
            
            report_data["question_results"].append({
                "question_num": question_num,
                "question": question,
                "answer": result["answer"],
                "context_retrieved": context_retrieved,
                "context_used": context_used,
                "context_details": result["context"],
                "timings": timings,
                "meve_raw_answer": result["metadata"]["meve_raw_answer"],
                "config": result["metadata"]["config"],
                "status": "success"
            })

            # Wait 1 second before next question
            if question_num < len(questions_to_process):
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n\n⏹️  Interrupted at question {question_num}. Generating report...\n")
            report_data["questions_processed"] = question_num - 1
            break
        except Exception as e:
            print(f"         ❌ Error: {str(e)}\n")
            report_data["questions_processed"] += 1
            report_data["failed"] += 1
            report_data["question_results"].append({
                "question_num": question_num,
                "question": question,
                "error": str(e),
                "status": "failed"
            })
            continue

    # Calculate statistics
    report_data["end_time"] = datetime.now().isoformat()
    report_data["duration_seconds"] = (datetime.now() - report_data["start_time"]).total_seconds()
    if report_data["successful"] > 0:
        report_data["avg_context_chunks"] = report_data["total_context_chunks_retrieved"] / report_data["successful"]
        report_data["avg_time_per_question"] = report_data["total_time_seconds"] / report_data["successful"]
    
    # Generate report
    generate_report(report_data, stats)

    print("\n" + "=" * 70)
    print(f"✅ Completed processing {report_data['questions_processed']} questions")
    print(f"📂 All results logged to: {stats['log_directory']}")
    print(f"📄 Report generated: report_nn.md")
    print("=" * 70 + "\n")


def generate_report(report_data: Dict[str, Any], system_stats: Dict[str, Any]) -> None:
    """Generate a comprehensive markdown report after processing all questions."""
    # Create result/all directory if it doesn't exist
    result_dir = Path("result/all")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract instance number from session ID (last part after the last underscore)
    session_id = report_data['session_id']
    instance_num = session_id.split('_')[-1][:6]  # Get first 6 chars of the last segment
    
    # Create report path with instance number
    report_path = result_dir / f"report_{instance_num}.md"
    
    with open(report_path, "w") as f:
        f.write("# MeVe RAG Batch Processing Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Session ID**: {report_data['session_id']}\n")
        f.write(f"- **Start Time**: {report_data['timestamp']}\n")
        f.write(f"- **End Time**: {report_data['end_time']}\n")
        f.write(f"- **Total Duration**: {report_data['duration_seconds']:.2f} seconds\n")
        f.write(f"- **Total Questions**: {report_data['total_questions']}\n")
        f.write(f"- **Questions Processed**: {report_data['questions_processed']}\n")
        f.write(f"- **Successful**: {report_data['successful']}\n")
        f.write(f"- **Failed**: {report_data['failed']}\n\n")
        
        # System Configuration
        f.write("## System Configuration\n\n")
        f.write(f"- **Collection Name**: {system_stats['collection_name']}\n")
        f.write(f"- **Total Chunks Available in DB**: {system_stats['total_chunks']}\n")
        f.write(f"- **Engine Status**: {'✅ Ready' if system_stats['engine_initialized'] else '❌ Failed'}\n")
        f.write(f"- **OpenAI Integration**: {'✅ Available' if system_stats['openai_initialized'] else '❌ Not Configured'}\n")
        f.write(f"- **Log Directory**: {system_stats['log_directory']}\n\n")
        
        # Processing Statistics
        f.write("## Processing Statistics\n\n")
        f.write(f"- **Total Context Chunks Retrieved**: {report_data['total_context_chunks_retrieved']}\n")
        f.write(f"- **Total Context Chunks Used**: {report_data['total_context_chunks_used']}\n")
        f.write(f"- **Average Chunks Retrieved per Question**: {report_data['avg_context_chunks']:.2f}\n")
        f.write(f"- **Average Chunks Used per Question**: {report_data['total_context_chunks_used'] / max(report_data['successful'], 1):.2f}\n")
        f.write(f"- **Success Rate**: {(report_data['successful'] / max(report_data['questions_processed'], 1) * 100):.1f}%\n")
        f.write(f"- **Total Processing Time**: {report_data['total_time_seconds']:.2f} seconds\n")
        f.write(f"- **Average Time per Question**: {report_data['avg_time_per_question']:.2f} seconds\n\n")
        
        # Summary Table
        f.write("## Quick Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Questions | {report_data['total_questions']} |\n")
        f.write(f"| Processed | {report_data['questions_processed']} |\n")
        f.write(f"| Success Rate | {(report_data['successful'] / max(report_data['questions_processed'], 1) * 100):.1f}% |\n")
        f.write(f"| Total Context Retrieved | {report_data['total_context_chunks_retrieved']} |\n")
        f.write(f"| Total Context Used | {report_data['total_context_chunks_used']} |\n")
        f.write(f"| Total Time | {report_data['total_time_seconds']:.2f}s |\n")
        f.write(f"| Avg Time per Question | {report_data['avg_time_per_question']:.2f}s |\n\n")
        
        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        f.write("| # | Question | Status | Retrieved | Used | Pipeline (s) | OpenAI (s) | Total (s) | Answer Preview |\n")
        f.write("|---|----------|--------|-----------|------|--------------|-----------|-----------|----------------|\n")
        
        for result in report_data["question_results"]:
            question_preview = result["question"][:45].replace("|", "\\|")
            status = "✅" if result["status"] == "success" else "❌"
            
            if result["status"] == "success":
                retrieved = result["context_retrieved"]
                used = result["context_used"]
                pipeline = result["timings"]["pipeline_seconds"]
                openai = result["timings"]["openai_seconds"]
                total = result["timings"]["total_seconds"]
                answer_preview = result["answer"][:35].replace("|", "\\|")
                f.write(f"| {result['question_num']} | {question_preview}... | {status} | {retrieved} | {used} | {pipeline:.3f} | {openai:.3f} | {total:.3f} | {answer_preview}... |\n")
            else:
                error_msg = result.get("error", "Unknown error")[:25].replace("|", "\\|")
                f.write(f"| {result['question_num']} | {question_preview}... | {status} | - | - | - | - | - | Error: {error_msg} |\n")
        
        f.write("\n")
        
        # Detailed Question Analysis
        f.write("## Detailed Question Analysis\n\n")
        
        for result in report_data["question_results"]:
            if result["status"] == "success":
                f.write(f"### Question {result['question_num']}\n\n")
                f.write(f"**Question**: {result['question']}\n\n")
                
                # Context Retrieved
                f.write(f"**Context Retrieved**: {result['context_retrieved']} chunks\n\n")
                
                if result['context_details']:
                    f.write("**Retrieved Context Details**:\n\n")
                    for i, ctx in enumerate(result['context_details'], 1):
                        f.write(f"#### Context Chunk {i}\n\n")
                        f.write(f"- **Document ID**: {ctx['doc_id']}\n")
                        f.write(f"- **Relevance Score**: {ctx.get('relevance_score', 'N/A')}\n")
                        f.write(f"- **Token Count**: {ctx.get('token_count', 'N/A')}\n")
                        f.write(f"- **Content**: {ctx['content'][:200]}...\n\n" if len(ctx['content']) > 200 else f"- **Content**: {ctx['content']}\n\n")
                
                # Timing Information
                f.write(f"**Timing Information**:\n\n")
                f.write(f"- **Pipeline Time**: {result['timings']['pipeline_seconds']:.3f} seconds\n")
                f.write(f"- **OpenAI Generation Time**: {result['timings']['openai_seconds']:.3f} seconds\n")
                f.write(f"- **Total Time**: {result['timings']['total_seconds']:.3f} seconds\n\n")
                
                # Configuration Used
                f.write(f"**Configuration Used**:\n\n")
                config = result['config']
                f.write(f"- **k_init**: {config.get('k_init', 'N/A')}\n")
                f.write(f"- **tau_relevance**: {config.get('tau_relevance', 'N/A')}\n")
                f.write(f"- **t_max**: {config.get('t_max', 'N/A')}\n\n")
                
                # MeVe Raw Answer
                f.write(f"**MeVe Raw Answer**: {result['meve_raw_answer']}\n\n")
                
                # Final AI-Generated Answer
                f.write(f"**AI-Generated Answer**: {result['answer']}\n\n")
                f.write("---\n\n")
        
        # Footer
        f.write("## Report Information\n\n")
        f.write(f"**Report Generated**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Session Logs Location**: {system_stats['log_directory']}\n\n")
        f.write("All detailed logs for individual Q&A pairs are stored in the session directory.\n")
    
    print(f"\n✅ Report saved to: {report_path}")


def interactive_mode():
    """Interactive Q&A CLI for the RAG system."""
    print("=" * 70)
    print("🤖 MeVe RAG Question-Answering System (Interactive Mode)")
    print("=" * 70)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Show statistics
    stats = rag.get_stats()
    print(f"\n📊 System Status:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Engine: {'✅ Ready' if stats['engine_initialized'] else '❌ Failed'}")
    print(f"   Logs: {stats['log_directory']}")

    if not stats["engine_initialized"]:
        print("\n❌ RAG system failed to initialize.")
        print("⚠️  Please ensure:")
        print("   • 'hotpotqa_contexts' collection exists in ChromaDB")
        print("   • Environment variables are set:")
        print("     - CHROMA_API_KEY")
        print("     - CHROMA_TENANT")
        print("     - CHROMA_DATABASE")
        return

    if not stats["openai_initialized"]:
        print("\n⚠️  OpenAI integration not available.")
        print("   Set OPENAI_API_KEY environment variable for natural language answers.")

    print("\n" + "=" * 70)
    print("💡 Ask any question and I'll search the knowledge base for answers!")
    print("   Type 'quit' or 'exit' to stop")
    print(f"   Logs saved to: {stats['log_directory']}\n")

    # Interactive Q&A loop
    question_count = 0
    while True:
        try:
            question = input("❓ Your question: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using MeVe RAG! Goodbye!\n")
                break

            question_count += 1
            print("\n🔍 Searching knowledge base...")

            # Get answer
            result = rag.answer_question(question)

            # Display answer
            print("\n" + "-" * 70)
            print(f"🤖 AI-GENERATED ANSWER (from OpenAI):")
            print("-" * 70)
            print(result["answer"])
            print("-" * 70)

            # Show retrieved context
            if result["context"]:
                print(f"\n📚 RETRIEVED CONTEXT ({len(result['context'])} chunks):")
                print("-" * 70)
                for i, chunk in enumerate(result["context"], 1):
                    print(f"\n[{i}] Document: {chunk['doc_id']}")
                    if chunk.get("relevance_score"):
                        print(f"    Score: {chunk['relevance_score']:.4f}")
                    print(f"    Content:\n{chunk['content']}")  # Full content, no truncation
            else:
                print("\n📚 No relevant context found in the knowledge base.")

            # Show metadata
            meta = result["metadata"]
            print("\n" + "-" * 70)
            print("📈 RETRIEVAL STATS:")
            print(
                f"   Retrieved: {meta.get('retrieved_chunks', 0)} / {meta.get('total_chunks', 0)} chunks"
            )
            print(f"   k_init: {meta.get('config', {}).get('k_init')}")
            print(f"   tau_relevance: {meta.get('config', {}).get('tau_relevance')}")
            print(f"   t_max: {meta.get('config', {}).get('t_max')} tokens")
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thank you for using MeVe RAG!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("-" * 70 + "\n")


def show_batch_mode_menu() -> tuple[int, List[str]]:
    """Display batch mode submenu and return option and questions list.
    
    Returns:
        Tuple of (option_choice, questions_list)
    """
    print("\n" + "=" * 70)
    print("🔄 MeVe RAG - Batch Mode Options")
    print("=" * 70)
    print("\nSelect batch processing mode:")
    print("  1️⃣  Loop through ALL questions")
    print("  2️⃣  Loop through RANDOM 10 questions")
    print("  3️⃣  Loop through custom number of RANDOM questions")
    print("  4️⃣  Back to main menu")
    print("\n" + "=" * 70)
    
    # Load all questions first
    try:
        with open("all_questions.json", "r") as f:
            all_questions = json.load(f)
    except FileNotFoundError:
        print("\n❌ Could not find all_questions.json")
        return 4, []
    
    while True:
        try:
            choice = input("\n📌 Enter your choice (1, 2, 3, or 4): ").strip()
            
            if choice == "1":
                print(f"\n✅ Selected: Process all {len(all_questions)} questions")
                return 1, all_questions
            elif choice == "2":
                selected = random.sample(all_questions, min(10, len(all_questions)))
                print(f"\n✅ Selected: Process {len(selected)} random questions")
                return 2, selected
            elif choice == "3":
                while True:
                    try:
                        num_str = input("\n📌 How many random questions do you want to process? ").strip()
                        num = int(num_str)
                        if num <= 0:
                            print("❌ Please enter a positive number.")
                            continue
                        if num > len(all_questions):
                            print(f"⚠️  Only {len(all_questions)} questions available. Processing all.")
                            num = len(all_questions)
                        selected = random.sample(all_questions, num)
                        print(f"\n✅ Selected: Process {len(selected)} random questions")
                        return 3, selected
                    except ValueError:
                        print("❌ Invalid number. Please enter a valid integer.")
            elif choice == "4":
                print("\n↩️  Returning to main menu...")
                return 4, []
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Returning to main menu...\n")
            return 4, []
        except Exception as e:
            print(f"❌ Error: {e}")


def show_menu() -> int:
    """Display the main menu and get user choice."""
    print("\n" + "=" * 70)
    print("🤖 MeVe RAG - Main Menu")
    print("=" * 70)
    print("\nSelect an option:")
    print("  1️⃣  Batch mode (process questions)")
    print("  2️⃣  Q&A mode (interactive)")
    print("  3️⃣  Exit")
    print("\n" + "=" * 70)
    
    while True:
        try:
            choice = input("\n📌 Enter your choice (1, 2, or 3): ").strip()
            if choice in ["1", "2", "3"]:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            return 3
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Show batch mode submenu
            batch_choice, selected_questions = show_batch_mode_menu()
            if batch_choice == 4:  # Back to main menu
                continue
            elif batch_choice in [1, 2, 3] and selected_questions:
                mode_name = ["all", "random 10", "custom random"][batch_choice - 1]
                print(f"\n🔄 Starting batch mode - looping through {mode_name} questions...\n")
                main(selected_questions)
        elif choice == 2:
            print("\n💬 Starting interactive Q&A mode...\n")
            interactive_mode()
        else:
            print("\n👋 Thank you for using MeVe RAG! Goodbye!\n")
            break
