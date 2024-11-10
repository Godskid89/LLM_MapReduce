import unittest
from llm_mapreduce.mapreduce import MapReduceLLM

# Sample long text for testing
sample_text = """Title: The Verdant Valley Ecosystem

                        In the heart of the continent lies the Verdant Valley, a unique ecosystem that stretches across hundreds of miles and is home to a wide variety of flora and fauna. The valley is characterized by its lush forests, winding rivers, expansive grasslands, and a mild climate that supports diverse species year-round.

                        The forested regions of the valley are dense with towering trees, including oaks, maples, and rare species of conifers that create a thick canopy overhead. Beneath the trees, a rich understory thrives, filled with ferns, mosses, and flowering plants that attract numerous pollinators. In the spring, the forest floor is carpeted with wildflowers, providing a burst of color and a critical food source for early-emerging insects and birds.

                        The rivers that weave through the valley are fed by snowmelt from the surrounding mountains. These rivers are teeming with fish such as trout and salmon, which migrate annually to spawn. The riverbanks are lined with willows and reeds, providing shelter for amphibians and nesting sites for a variety of birds. Beavers play a key role in the ecosystem by building dams that create ponds and wetlands, which in turn support diverse aquatic life, including frogs, turtles, and waterfowl.

                        Expansive grasslands stretch across the central part of the valley, providing grazing areas for large herbivores such as deer, elk, and wild horses. These grasslands are dotted with wildflowers and shrubs, which provide cover and food for smaller animals, including rabbits, foxes, and various rodents. Predators, including wolves, mountain lions, and birds of prey, patrol the grasslands, maintaining the balance of the ecosystem by keeping herbivore populations in check.

                        Throughout the year, the valley experiences a range of climatic conditions that shape the behavior and survival strategies of its inhabitants. The mild winters are marked by occasional snowfall, which blankets the valley in white but quickly melts away, allowing plants to grow almost year-round. Summers bring a warm, temperate climate with regular rainfall, nourishing the valley and replenishing its rivers. Autumn is a time of abundance, as animals prepare for the leaner winter months by feasting on the valley’s bountiful resources.

                        Human activities have had an impact on the Verdant Valley ecosystem over the years. The introduction of agriculture has led to the clearing of some grasslands for crops, affecting the habitats of many species. Additionally, the presence of human settlements along the rivers has introduced pollutants and disrupted the natural flow of water, impacting fish populations and the overall health of aquatic systems. Conservation efforts have been implemented in recent years to address these challenges. Organizations are working to restore habitats, reintroduce native species, and establish protected areas to preserve the natural beauty and biodiversity of the valley for future generations.

                        The Verdant Valley ecosystem serves as a reminder of the intricate balance that exists in nature. Each species, from the smallest insect to the largest predator, plays a role in maintaining the health and resilience of the environment. The valley’s forests, rivers, and grasslands are interconnected systems that support each other, forming a complex web of life that is both beautiful and fragile. Protecting such ecosystems is essential not only for the plants and animals that live there but also for the human communities that rely on the natural world for inspiration, recreation, and resources."""

# MockModel for testing
class MockModel:
    def generate(self, query):
        return {
            "text": f"Mocked response for query: {query}",
            "rationale": "Mock rationale based on provided text",
            "answer": f"Mock answer based on query: {query}"
        }

class TestMapReduceLLM(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.mapreduce = MapReduceLLM(model=self.model, context_window=512)

    def test_summarize_ecosystem(self):
        query = "Summarize the key points about the Verdant Valley ecosystem."
        result = self.mapreduce.process_long_text(sample_text, query)
        expected_phrase = f"Mock answer based on query: {query}"
        self.assertIn(expected_phrase, result["answer"])


    def test_flora_fauna_list(self):
        query = "List the types of flora and fauna found in the Verdant Valley."
        result = self.mapreduce.process_long_text(sample_text, query)
        self.assertIn("flora", result["answer"].lower())

    def test_human_impact(self):
        query = "What are the effects of human activities on the Verdant Valley ecosystem?"
        result = self.mapreduce.process_long_text(sample_text, query)
        self.assertIn("human activities", result["answer"].lower())

    def test_rivers_role(self):
        query = "Describe the role of rivers in the Verdant Valley ecosystem."
        result = self.mapreduce.process_long_text(sample_text, query)
        self.assertIn("rivers", result["answer"].lower())

    def test_seasonal_changes(self):
        query = "Explain how the seasons affect the Verdant Valley ecosystem."
        result = self.mapreduce.process_long_text(sample_text, query)
        self.assertIn("seasons", result["answer"].lower())

if __name__ == "__main__":
    unittest.main()