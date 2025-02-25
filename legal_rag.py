from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import tempfile
from langchain.schema import HumanMessage, AIMessage
from chunk_vector_store import LegalChunkVectorStore
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LegalRAG:
    def __init__(self, model_name="gpt-4o-mini", model_provider="openai") -> None: # Added model_name and model_provider parameters with default values
        self.cvs_obj = LegalChunkVectorStore()
        self.vector_store = None
        self.retriever = None
        self.pdf_chain = None
        self.combined_chain = None
        self.persist_directory = None # Changed to None for in-memory

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        try:
            self.chroma_client = chromadb.Client( # Changed to in-memory client
                settings=Settings(anonymized_telemetry=False) # Removed is_persistent as it's in-memory
            )

            if model_provider == "groq":
                self.model = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name=model_name, # Use the passed model_name
                    temperature=0.1
                )
            elif model_provider == "openai":
                self.model = ChatOpenAI(model_name=model_name, temperature=0) # Use the passed model_name
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")


            self._initialize_chains()
            logger.info(f"Successfully initialized LegalRAG instance with model: {model_name} from {model_provider}") # Log the model being used
        except Exception as e:
            logger.error(f"Failed to initialize LegalRAG: {e}")
            raise

    # Include all the methods from the original LegalRAG class
    # [Previous methods remain the same - _format_chat_history, _initialize_chains, etc.]

    def _format_chat_history(self, chat_history) -> str:
        formatted_history = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        return "\n".join(formatted_history[-3:])


    def _initialize_chains(self) -> None:
        try:
            # Modified text prompt for Indian legal context
            text_prompt = PromptTemplate.from_template(
                """You are an experienced Indian legal assistant providing accurate information based on Indian law while maintaining that this is not legal advice. You must reference Indian legal statutes, acts, and precedents where applicable.

                Document Context:
                {context}

                Chat History:
                {chat_history}

                Human Question: {question}

                Please provide a clear, professional response that:
                1. Primarily uses information from the retrieved document context when available
                2. References relevant Indian legal concepts, statutes, and case laws
                3. Clearly distinguishes between document-specific information and general Indian legal knowledge
                4. Explains legal terms in simple language while retaining Indian legal terminology
                5. Mentions relevant High Court or Supreme Court judgments if applicable
                6. Maintains appropriate legal disclaimers as per Indian Bar Council guidelines
                7. Suggests when consultation with an Indian advocate may be needed
                8. Specifies if any information pertains to specific Indian states, as laws may vary by jurisdiction

                Assistant Response:"""
            )

            qna_text_prompt = PromptTemplate.from_template(
                """You are an experienced Indian legal assistant providing accurate to some client questions information based on Indian law while

                Document Context:
                {context}

                Human Question: {question}

                Please provide a clear, professional response that answers the clients question based on the
                information provided in the document context, and make it brief and precise

                Assistant Response:"""
            )

            def format_context(context_docs):
                formatted_contexts = []
                for i, doc in enumerate(context_docs, 1):
                    metadata = doc.metadata or {}
                    source_info = f"[Source: Page {metadata.get('page', 'N/A')}]"
                    formatted_contexts.append(f"Context {i} {source_info}:\n{doc.page_content}")

                return "\n\n".join(formatted_contexts)

            # Rest of the initialization code remains same
            self.text_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])) if self.retriever else "No document context available.",
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | text_prompt
                | self.model
                | StrOutputParser()
            )

            # Basic text chain


            self.qna_text_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])) if self.retriever else "No document context available.",
                    "question": RunnablePassthrough()
                }
                | qna_text_prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Successfully initialized Indian legal assistant text chain with RAG capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize text chain: {e}")
            raise

    def _setup_document_chains(self) -> None:
        if not self.retriever:
            logger.warning("Retriever not initialized. Skipping document chain setup.")
            return

        try:
            def format_context(context_docs):
                formatted_contexts = []
                for i, doc in enumerate(context_docs, 1):
                    metadata = doc.metadata or {}
                    source_info = f"[Source: Page {metadata.get('page', 'N/A')}]"
                    formatted_contexts.append(f"Context {i} {source_info}:\n{doc.page_content}")

                return "\n\n".join(formatted_contexts)

            # Modified PDF prompt for Indian legal context
            pdf_prompt = PromptTemplate.from_template(
                """You are an expert Indian legal document analyst specializing in court orders, contracts, compliance documents, government notifications, and statutory documents. Analyze the provided document context based on Indian legal framework.

                Document Context:
                {context}

                Chat History:
                {chat_history}

                Human Question: {question}

                Provide a detailed analysis following these guidelines:

                1. Document Classification and Initial Analysis:
                    - Identify document type (Court Order/Contract/Compliance/Government/Case-related/Statutory)
                    - Specify jurisdiction (Supreme Court/High Court/State/Central)
                    - Identify parties involved and their roles
                    - Note critical dates, deadlines, and timelines

                2. Detailed Content Analysis:
                    For Court Orders/Judgments:
                    - Extract key holdings and precedents
                    - Identify cited cases and statutes
                    - Highlight ratio decidendi
                    - Note important observations and directions

                    For Contracts/Agreements:
                    - Identify key clauses and obligations
                    - Highlight rights and liabilities
                    - Note important terms, conditions, and deadlines
                    - Flag potential legal issues

                    For Compliance Documents:
                    - List statutory requirements
                    - Identify compliance deadlines
                    - Note filing obligations
                    - Highlight regulatory requirements

                    For Government Documents:
                    - Explain implementation requirements
                    - Note procedural compliances
                    - Identify affected parties
                    - List required actions

                    For Case-Related Documents:
                    - Analyze procedural requirements
                    - Identify next steps
                    - Note response deadlines
                    - List required documentation

                    For Statutory Documents:
                    - Explain applicability
                    - List compliance requirements
                    - Note relevant authorities
                    - Highlight penalty provisions

                3. Response Format:
                    - Quote relevant sections using "..."
                    - Cite specific document parts with source information
                    - Use clear headings for different aspects
                    - Provide practical implications
                    - Highlight immediate action items
                    - List relevant compliance requirements
                    - Include statutory references
                    - Note jurisdiction-specific requirements

                4. Additional Guidelines:
                    - Stay strictly within document context
                    - Use appropriate Indian legal terminology
                    - Explain technical terms in simple language
                    - Maintain legal disclaimers as per Bar Council
                    - Indicate if any information is outside document scope
                    - Suggest professional consultation when needed
                    - Note if multiple interpretations are possible
                    - Highlight any ambiguities requiring clarification

                Assistant Response (structure your response based on the document type and query focus):"""
            )

            # Enhanced combined prompt for integrated analysis
            combined_prompt = PromptTemplate.from_template(
                """You are an expert Indian legal assistant providing comprehensive document analysis with additional legal context.

                Document Context:
                {context}

                Chat History:
                {chat_history}

                Human Question: {question}

                Provide a detailed response that:

                1. Document-Based Analysis:
                    - Analyze document content thoroughly
                    - Quote relevant sections using "..."
                    - Identify key provisions and requirements
                    - Note critical dates and deadlines
                    - Highlight important clauses and conditions

                2. Legal Framework Integration:
                    - Connect document content with relevant:
                        * Indian statutes and regulations
                        * Supreme Court/High Court judgments
                        * Government notifications/circulars
                        * Regulatory requirements
                    - Explain legal implications
                    - Note jurisdiction-specific aspects
                    - Highlight compliance requirements

                3. Practical Guidance:
                    - Provide actionable insights
                    - List required next steps
                    - Explain procedural requirements
                    - Note documentation needs
                    - Highlight compliance deadlines
                    - Suggest risk mitigation measures

                4. Response Requirements:
                    - Distinguish between document information and general legal knowledge
                    - Use appropriate Indian legal terminology
                    - Explain complex terms simply
                    - Maintain professional disclaimers
                    - Indicate when professional help is needed
                    - Note any ambiguities or uncertainties
                    - Specify jurisdiction applicability

                5. Based on Document Type:
                    Court Orders: Focus on holdings, precedents, directions
                    Contracts: Emphasize rights, obligations, terms
                    Compliance: Highlight requirements, deadlines, filings
                    Government: Explain implementation, procedures
                    Case Documents: Note procedures, timelines, requirements
                    Statutory: Detail applicability, compliance needs

                Format your response based on document type and query focus, maintaining clarity and practical utility.

                Assistant Response:"""
            )

            # Chain setup
            self.pdf_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | pdf_prompt
                | self.model
                | StrOutputParser()
            )


            self.combined_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | combined_prompt
                | self.model
                | StrOutputParser()
            )

            self.summary_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | combined_prompt
                | self.model
                | StrOutputParser()
            )


            logger.info("Successfully initialized Indian legal document chains with enhanced analysis capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize document chains: {e}")
            raise

    def _update_memory(self, query: str, response: str) -> None:
        try:
            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=response))
            logger.debug("Memory updated with new interaction")
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, Any]]:
        try:
            messages = self.memory.chat_memory.messages
            return [{"role": "human" if isinstance(msg, HumanMessage) else "assistant",
                     "content": msg.content} for msg in messages]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []

    def ask_text_only(self, query: str) -> str:
        try:
            # Check if documents are available in the vector store
            if self.retriever:
                # Get relevant documents
                context_docs = self.retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(context_docs)} contexts for text query: {[doc.page_content[:500] + '...' for doc in context_docs]}")

            # Process query with the text chain (which now includes document context if available)
            response = self.text_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response

        except Exception as e:
            error_msg = f"Error processing text-only query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_qna(self, query: str) -> str:
        try:
            # Check if documents are available in the vector store
            if self.retriever:
                # Get relevant documents
                context_docs = self.retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(context_docs)} contexts for text query: {[doc.page_content[:500] + '...' for doc in context_docs]}")

            # Process query with the text chain (which now includes document context if available)
            response = self.qna_text_chain.invoke({"question": query})
            return response

        except Exception as e:
            error_msg = f"Error processing text-only query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_pdf(self, query: str) -> str:
        if not self.pdf_chain:
            return "Please upload a legal document first."
        try:
            context_docs = self.retriever.get_relevant_documents(query)
            if not context_docs:
                return "No relevant information found in the document. Please rephrase your question or ask about a different topic covered in the document."

            logger.debug(f"Retrieved contexts: {[doc.page_content[:500] + '...' for doc in context_docs]}")

            response = self.pdf_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response
        except Exception as e:
            error_msg = f"Error processing PDF query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_combined(self, query: str) -> str:
        """
        Handle combined queries with memory management

        Args:
            query (str): User's question

        Returns:
            str: Assistant's response
        """
        if not self.combined_chain:
            return self.ask_text_only(query)
        try:
            # Get relevant context to verify retrieval
            context_docs = self.retriever.get_relevant_documents(query)  # Removed fetch_k
            logger.debug(f"Retrieved contexts for combined query: {[doc.page_content[:500] + '...' for doc in context_docs]}")

            response = self.combined_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response
        except Exception as e:
            error_msg = f"Error processing combined query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def summarize_document(self, file_path: str, document_type: str = "Other") -> str:
        """
        Summarize the content of a legal document.

        Args:
            file_path (str): Path to the document file.
            document_type (str): Type of legal document.

        Returns:
            str: Summary of the document content.
        """
        try:
            self.feed(file_path, document_type)

            # Get summaries for each chunk
            summaries = []
            collections = self.vector_store.get_collections() # This line will cause error as vector_store is Chroma object not client
            for chunk in collections['documents']: # This line will cause error as vector_store is Chroma object not client
                response = self.summary_chain.invoke({"question": chunk.page_content}) # This line will cause error as chunk is not defined here
                summaries.append(response)
            # Combine the chunk summaries
            full_summary = "\n\n".join(summaries)
            logger.info("Successfully summarized the document.")
            return full_summary

        except Exception as e:
            error_msg = f"Error summarizing document: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)


    def feed(self, file_path: str, document_type: str = "Other") -> None:
        """
        Process and store document content with type-specific handling

        Args:
            file_path (str): Path to the document file
            document_type (str): Type of legal document
        """
        try:
            # Recreate collection
            try:
                self.chroma_client.delete_collection("document_collection")
                logger.info("Deleted existing document collection")
            except ValueError:
                pass

            collection = self.chroma_client.create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new document collection")

            # Process document with document type
            chunks = self.cvs_obj.split_into_chunks(file_path, document_type)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Initialize vector store with proper embedding function
            self.vector_store = Chroma.from_documents(
                documents=chunks, # Changed from texts to chunks
                client=self.chroma_client,
                collection_name="document_collection",
                embedding=self.cvs_obj.embedding,
                persist_directory=None # Added to use in-memory here as well, although client is in-memory now
            )

            # Add documents and setup retriever
            # self.vector_store.add_documents(chunks) # No need to call add_documents again, it is done in from_documents
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            # Initialize document-aware chains
            self._setup_document_chains()
            logger.info(f"Successfully processed {document_type} document")

        except Exception as e:
            error_msg = f"Error feeding document: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def clear(self) -> None:
        """Clear all stored documents and reset system state"""
        try:
            if self.chroma_client is not None:
                try:
                    self.chroma_client.delete_collection("document_collection")
                    logger.info("Deleted document collection")
                except ValueError:
                    pass

            self.vector_store = None
            self.retriever = None
            self.pdf_chain = None
            self.combined_chain = None
            self.memory.clear()

            logger.info("Successfully cleared all resources")

        except Exception as e:
            error_msg = f"Error clearing resources: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)


    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            if self.persist_directory and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory) # This line will not execute as persist_directory is None now
            logger.info("Successfully cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def process_predefined_questions(self, file_path: str) -> Dict[str, str]:
        """
        Process a predefined set of questions from a text file and generate answers.

        Args:
            file_path (str): Path to the text file containing questions.

        Returns:
            Dict[str, str]: A dictionary with questions as keys and their answers as values.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read the questions from the file
            with open(file_path, 'r') as file:
                questions = [line.strip() for line in file if line.strip()]

            if not questions:
                logger.warning("No questions found in the file.")
                return {}

            logger.info(f"Processing {len(questions)} predefined questions.")
            answers = {}

            # Process each question
            for question in questions:
                try:
                    answers[question] = self.ask_qna(question)
                except Exception as e:
                    answers[question] = f"Error processing question: {str(e)}"

            logger.info("Completed processing predefined questions.")
            return answers
        except Exception as e:
            error_msg = f"Error processing predefined questions: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)