
from dotenv import load_dotenv
from crewai import Crew
from tasks import ContentGenerationTasks
from agents import ContentGenerationAgents
from tools import Extraction_Tools

def main():

    load_dotenv()


    print("## Welcome to the Content Generation Crew")
    print('---------------------------------')

    content_prompt = input("Please provide a prompt for the new writing content you would like to be generated, specifying type of content (blogpost, linkedIn post, etc.), and any specific word count if you would like: \n")
    context = input("Please provide the url to any writing pieces you would like to provide as stylistic context: \n")
    context_file = Extraction_Tools.url_to_md(context)

    
    
    tasks = ContentGenerationTasks()
    agents = ContentGenerationAgents()

    extraction_agent = agents.extraction_agent()
    prompt_analyst_agent = agents.prompt_analyst_agent()
    researcher_agent = agents.researcher_agent()
    writer_agent = agents.writer_agent()
    humanizer_agent = agents.humanizer_agent()

    extraction_task = tasks.extraction_task(extraction_agent, context_file)
    analyze_prompt_task = tasks.analyze_prompt_task(prompt_analyst_agent, content_prompt)
    researcher_task = tasks.research_task(researcher_agent)
    writer_task = tasks.writer_task(writer_agent)
    humanizer_task = tasks.humanizer_task(humanizer_agent)

    researcher_task.context = [analyze_prompt_task]
    writer_task.context = [extraction_task, analyze_prompt_task, researcher_task]
    humanizer_task.context = [writer_task]

    
    content_generation_crew = Crew(

        agent = [
            extraction_agent,
            prompt_analyst_agent,
            researcher_agent,
            writer_agent,
            humanizer_agent
        ],
        tasks = [
            extraction_task,
            analyze_prompt_task,
            researcher_task,
            writer_task,
            humanizer_task
        ]
    )

    result = content_generation_crew.kickoff()

    output = humanizer_task.output.raw_output

    print(output)


if __name__ == "__main__":
    main()
