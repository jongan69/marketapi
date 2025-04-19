# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")

def estimate_sentiment(news):
    if news:
        # Handle both single string and list of strings
        if isinstance(news, str):
            news = [news]
        
        # Get predictions from pipeline
        results = pipe(news)
        
        # Process results
        if len(news) == 1:
            # Single text case
            result = results[0]
            return result['score'], result['label']
        else:
            # Multiple texts case
            # Average the scores for each label
            label_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            for result in results:
                label_scores[result['label']] += result['score']
            
            # Get the label with highest average score
            max_label = max(label_scores, key=label_scores.get)
            return label_scores[max_label], max_label
    else:
        return 0, "neutral"


if __name__ == "__main__":
    # Test with single text
    score, sentiment = estimate_sentiment('markets responded negatively to the news!')
    print(f"Single text - Score: {score}, Sentiment: {sentiment}")
    
    # Test with multiple texts
    score, sentiment = estimate_sentiment(['markets responded negatively to the news!', 'traders were displeased!'])
    print(f"Multiple texts - Score: {score}, Sentiment: {sentiment}")
