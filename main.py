from agent import process_users
import asyncio

# For local testing
async def deploy_agent():
    """
    Main function. All agent calls will occur here.
    """
    await process_users()

# For local testing
if __name__ == '__main__':
    asyncio.run(deploy_agent())



