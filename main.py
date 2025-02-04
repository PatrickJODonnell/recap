from agent import invoke_agent
import asyncio

# Todo - add prefect decorator and install
async def deploy_agent():
    """
    Main function. All agent calls will occur here.
    """
    await invoke_agent()

# For local testing
if __name__() == '__main__':
    asyncio.run(deploy_agent)



