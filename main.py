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

# TODO - ERROR HANDLE ANYTHING THAT COMES UP DURING EXECUTION
# TODO - LOOK INTO HOW TO HOST THIS THING SO IT RUNS ON CLOUD COMPUTE INSTEAD OF MY MACHINE
# TODO - LOOK INTO DIFFERENT METHOD OF RUNNING -> HAVING ALL USERS AND ALL OF THOSE LOCAL AUDIO FILES WILL GET IMPOSSIBLE TO LOAD INTO MEMORY



