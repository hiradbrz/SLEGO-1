from pyvis.network import Network
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import openai

class KnowledgeGraphDrawer:
    def __init__(self, knowledge_dict, output_file='knowledge_graph.html'):
        self.knowledge_dict = knowledge_dict
        self.output_file = output_file
        self.colors = ['red', 'green', 'blue']
        self.create_graph()

    def create_graph(self):
        G = Network(notebook=True, directed=True)

        first_layer_nodes = list(self.knowledge_dict.keys())
        for i, node in enumerate(first_layer_nodes):
            G.add_node(node, label=node, title=str(self.knowledge_dict[node]), color=self.colors[0])

        for i, node in enumerate(first_layer_nodes):
            for j, (attribute, value) in enumerate(self.knowledge_dict[node].items()):
                attribute_node = f"{node}_{attribute}"
                G.add_node(attribute_node, label=attribute, color=self.colors[1])
                G.add_edge(attribute_node, node, arrows='to')
                value_node = f"{attribute_node}_{value}"
                G.add_node(value_node, label=str(value), color=self.colors[2])
                G.add_edge(value_node, attribute_node, arrows='to')

        for i in range(len(first_layer_nodes) - 1):
            G.add_edge(first_layer_nodes[i], first_layer_nodes[i+1], arrows='to')

        OPTIONS  = {
            "solver": "forceAtlas2Based",
            "physics": {
                "gravitationalConstant": -300,
                "springLength": 200,
                "springConstant": 0.05
            }
        }
        G.set_options(json.dumps(OPTIONS))

        G.show(self.output_file)

class KnowledgeGraphRecommender:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.vectorizer = CountVectorizer()
        self.task_vectors = self._create_task_vectors()

    def _create_task_vectors(self):
        tasks = [" ".join([str(val) for val in task.values()]) for task in self.knowledge_graph.values()]
        return self.vectorizer.fit_transform(tasks)

    def _compare_tasks(self, new_task):
        new_task_str = " ".join([str(val) for val in new_task.values()])
        new_task_vec = self.vectorizer.transform([new_task_str])  # Use the same vectorizer
        similarities = cosine_similarity(new_task_vec, self.task_vectors)
        return np.argmax(similarities), np.max(similarities)

    def recommend_pipeline(self, new_json):
        recommendations = defaultdict(list)
        for task_name, task_attrs in new_json.items():
            most_similar_task_idx, similarity = self._compare_tasks(task_attrs)
            if similarity > 0.5:  # You can adjust this threshold
                similar_task = list(self.knowledge_graph.keys())[most_similar_task_idx]
                recommendations[similar_task].append((task_name, similarity))
        return recommendations

    def create_updated_pipeline(self, new_json):
        new_pipeline = self.knowledge_graph.copy()
        recommendations = self.recommend_pipeline(new_json)

        # Replace entire tasks based on recommendations
        for existing_task, replacements in recommendations.items():
            for new_task, _ in replacements:
                if new_task in new_json:
                    # Replace the task but maintain the original 'goal'
                    original_goal = new_pipeline[existing_task].get('goal')
                    new_pipeline[existing_task] = new_json[new_task]
                    if original_goal:
                        new_pipeline[existing_task]['goal'] = original_goal

        # Detect and store the new ticker
        new_ticker = self._extract_new_attribute(new_json, 'ticker')

        # Detect and store the old ticker
        old_ticker = self._extract_old_attribute('ticker')

        # Apply ticker changes dynamically across the pipeline
        if new_ticker and old_ticker:
            self._update_ticker_related_fields(new_pipeline, old_ticker, new_ticker)

        return json.dumps(new_pipeline, indent=4)

    def _extract_new_attribute(self, new_json, attribute):
        for task_details in new_json.values():
            if attribute in task_details:
                return task_details[attribute]
        return None

    def _extract_old_attribute(self, attribute):
        for task_details in self.knowledge_graph.values():
            if attribute in task_details:
                return task_details[attribute]
        return None

    def _update_ticker_related_fields(self, pipeline, old_ticker, new_ticker):
        for task, details in pipeline.items():
            for key in details:
                if isinstance(details[key], str):
                    details[key] = details[key].replace(old_ticker, new_ticker)

def generate_change_summary(old_pipeline, new_pipeline, graph_db_pipeline):
    """
    Generates a summary of the changes made to the pipeline using GPT-3.5.

    :param old_pipeline: The original pipeline.
    :param new_pipeline: The updated pipeline.
    :param graph_db_pipeline: The pipeline from the graph database.
    :return: A string summary of the changes.
    """

    # Format the context for GPT-3.5
    prompt = f"Context: A data processing pipeline has been updated based on new requirements. " \
                f"Here are the original pipeline, the new changes, and the reference pipeline from a graph database. " \
                f"Generate a clear and concise summary of the changes made, highlighting any discrepancies with the graph database pipeline.\n\n" \
                f"Original Pipeline:\n{json.dumps(old_pipeline, indent=2)}\n\n" \
                f"New Pipeline:\n{json.dumps(new_pipeline, indent=2)}\n\n" \
                f"Graph Database Pipeline:\n{json.dumps(graph_db_pipeline, indent=2)}\n\n" \
                f"Summary:"
    
    openai.api_key = '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'
    # Call to GPT-3.5 API (this is a placeholder - replace with actual API call)
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )

    # Extract and return the summary from the response
    return response.choices[0].text.strip()


# Example usage:
# Original JSON for Knowledge Graph
json_data = {
    "get_stock_data_101": {"ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2021-01-01", "output_s3_file_key": "data/AAPL_stock_data.csv", "goal": "plot stock data"},
    "preprocess_stock_data_101": {"input_s3_file_key": "data/AAPL_stock_data.csv", "output_s3_file_key": "data/AAPL_stock_data_processed.csv", "goal": "plot stock data"},
    "compute_simple_moving_average_101": {"input_s3_file_key": "data/AAPL_stock_data_processed.csv", "window_size": 20, "output_s3_file_key": "data/AAPL_stock_data_SMA.csv", "goal": "plot stock data"},
    "plot_stock_data_101": {"input_s3_file_key": "data/AAPL_stock_data_SMA.csv", "ticker": "AAPL", "output_html_file_key": "plots/AAPL_stock_plot.html", "goal": "plot stock data"}
}
# New JSON data for recommendation
new_json_data = {
    "fetch_new_stock_data": {"ticker": "MSFT", "start_date": "2022-01-01", "end_date": "2023-01-01", "output_s3_file_key": "data/MSFT_stock_data.csv", "goal": "analyze stock data"},
    "analyze_stock_performance": {"input_s3_file_key": "data/MSFT_stock_data.csv", "output_s3_file_key": "data/MSFT_stock_analysis.csv", "goal": "analyze stock data"}
}


knowledge_graph_drawer = KnowledgeGraphDrawer(json_data, 'knowledge_graph2.html')
recommender = KnowledgeGraphRecommender(json_data)
recommendations = recommender.recommend_pipeline(new_json_data)
updated_pipeline_json = recommender.create_updated_pipeline(new_json_data)
summary = generate_change_summary(new_json_data,updated_pipeline_json,json_data)

print(recommendations)
print(updated_pipeline_json)
print(summary)
