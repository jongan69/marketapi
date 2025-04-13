from typing import Optional, List
import asyncio
from textwrap import wrap
import dotenv
from g4f.client import Client

dotenv.load_dotenv(override=True)    

class FOMCSummarizer:
    """Class to handle chunking and summarizing FOMC minutes text using AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the summarizer with g4f client."""
        # No API key needed for g4f
        self.client = Client()
    
    async def summarize_text(self, text: str, chunk_size: int = 1500) -> str:
        """
        Summarize FOMC minutes text by chunking it and using g4f.
        
        Args:
            text: The FOMC minutes text to summarize
            chunk_size: Maximum size of each chunk in characters
            
        Returns:
            A summarized version of the text
        """
        if not text or len(text.strip()) < 100:
            return "No sufficient text to summarize."
        
        # Split text into chunks
        chunks = self._chunk_text(text, chunk_size)
        
        # Process chunks in parallel
        chunk_summaries = await asyncio.gather(
            *[self._summarize_chunk(chunk, i+1, len(chunks)) for i, chunk in enumerate(chunks)]
        )
        
        # Combine summaries
        combined = "\n\n".join(chunk_summaries)
        
        # Final summarization
        final_summary = await self._create_final_summary(combined)
        
        return final_summary
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately equal size."""
        # Use textwrap to split text into chunks
        chunks = wrap(text, chunk_size)
        
        # If we have too many chunks, increase the chunk size
        if len(chunks) > 10:
            return self._chunk_text(text, chunk_size * 2)
        
        return chunks
    
    async def _summarize_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """Summarize a single chunk of text."""
        try:
            # Run the g4f call in a thread pool since it's not async
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Federal Reserve monetary policy and FOMC meetings. Your task is to summarize FOMC minutes accurately and concisely."},
                    {"role": "user", "content": f"Summarize this portion (part {chunk_num} of {total_chunks}) of the FOMC minutes, focusing on key policy decisions, economic assessments, and important discussions:\n\n{chunk}"}
                ],
                web_search=False
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing chunk {chunk_num}: {e}")
            return f"[Error summarizing part {chunk_num} of {total_chunks}]"
    
    async def _create_final_summary(self, combined_summaries: str) -> str:
        """Create a final summary from the combined chunk summaries."""
        try:
            # Run the g4f call in a thread pool since it's not async
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Federal Reserve monetary policy and FOMC meetings. Your task is to create a concise, accurate summary of FOMC minutes."},
                    {"role": "user", "content": f"Create a comprehensive but concise summary of the FOMC minutes from these section summaries. Focus on key policy decisions, economic assessments, and important discussions. Format the summary with clear sections for 'Key Decisions', 'Economic Assessment', and 'Policy Outlook':\n\n{combined_summaries}"}
                ],
                web_search=False
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error creating final summary: {e}")
            return "Error creating final summary."

# Example usage
async def summarize_fomc_minutes(minutes_text: str, api_key: Optional[str] = None) -> str:
    """
    Convenience function to summarize FOMC minutes text.
    
    Args:
        minutes_text: The FOMC minutes text to summarize
        api_key: Not used with g4f
        
    Returns:
        A summarized version of the minutes
    """
    summarizer = FOMCSummarizer()
    return await summarizer.summarize_text(minutes_text) 