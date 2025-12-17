from typing import List, Optional
from langchain_core.documents import Document
from src.config import SUMMARIZER_SYSTEM_INSTRUCTION, STUDY_GUIDE_SYSTEM_INSTRUCTIONS

class Summarizer:
    def __init__(self,vector_store,generator,max_chars_per_batch: int = 5000):
        self.vector_store = vector_store               #ChromaStore
        self.generator = generator                     #Generator
        self.max_chars = max_chars_per_batch

    def _fetch_docs(self, where: Optional[dict] = None) -> List[Document]:
        return self.vector_store.get_all_documents(where)
        # made a function to get all docs from chroma_store    

    def _batch_docs(self, docs: List[Document]) -> List[str]:
        batches = []
        current = ""

        for doc in docs:
            text = doc.page_content.strip()
            if len(current) + len(text) > self.max_chars:
                batches.append(current)
                current = text
            elif not current:
                current=text
            else:
                current += "\n\n" + text

        if current.strip():
            batches.append(current)

        return batches

    def _summarize_batch(self, text: str, mode: str, client=None, model_name: str = None, ollama_model: str = None,) -> str:

        prompt = f"""
Material:
{text}
Create a concise bullet-point summary of the following material.
Focus on key concepts and definitions.
""".strip()

        if mode == "ONLINE":
            return self.generator.generate_online(
                prompt=prompt,
                client=client,
                model_name=model_name,
                system_instruction = SUMMARIZER_SYSTEM_INSTRUCTION,
            )

        return self.generator.generate_offline(
            prompt=prompt,
            model=ollama_model,
            system_instruction = SUMMARIZER_SYSTEM_INSTRUCTION,
        )

    # --------------------------------------------------
    # generate study guide
    # --------------------------------------------------
    def generate_study_guide(
            self,
            mode: str,
            client=None,
            model_name: str = None,
            ollama_model: str = None,
            where: Optional[dict] = None,
            progress_callback=None,               # added so we can see which batch is being loaded in the frontend
            ) -> str:
        
        docs = self._fetch_docs(where=where)
        docs = sorted(
            docs,
            key=lambda d: (
                d.metadata.get("file_name", ""),
                d.metadata.get("page", 0),
            )
        )

        if not docs:
            return "No material available to summarize."

        batches = self._batch_docs(docs)

        partial_summaries = []
        for idx, batch in enumerate(batches):
            if progress_callback:
                progress_callback(f"Summarizing batch {idx+1}/{len(batches)}")

            try:
                summary = self._summarize_batch(
                    batch,
                    mode=mode,
                    client=client,
                    model_name=model_name,
                    ollama_model=ollama_model,
                )
            except Exception as e:
                summary = f"[Summary failed for batch {idx + 1}: {e}]"
            
            partial_summaries.append(summary)


        # Final merge pass
        merge_prompt = f"""
You are a study assistant.

Combine the following summaries into a structured study guide.
Use headings and bullet points.
Do NOT add new information.

Summaries:
{"\n\n".join(partial_summaries)}

Final Study Guide:
""".strip()

        if mode == "ONLINE":
            return self.generator.generate_online(
                prompt=merge_prompt,
                client=client,
                model_name=model_name,
                system_instruction = STUDY_GUIDE_SYSTEM_INSTRUCTIONS,
            )

        return self.generator.generate_offline(
            prompt=merge_prompt,
            model=ollama_model,
            system_instruction = STUDY_GUIDE_SYSTEM_INSTRUCTIONS,
        )


