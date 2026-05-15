from app.orchestrator.workflow import research


if __name__ == "__main__":

    topic = input("Enter research topic: ")

    research(topic)