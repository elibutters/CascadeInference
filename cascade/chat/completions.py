import asyncio
import functools
from cascade.strategies import StrictAgreement, SemanticAgreement

STRATEGY_MAPPING = {
    "strict": StrictAgreement,
    "semantic": SemanticAgreement,
}

async def create(level1_clients, level2_client, agreement_strategy, messages, **kwargs):
    """
    This is the main function for Cascade Inference.
    It performs level 1 inference calls asynchronously and prepares for comparison.
    """
    
    tasks = []
    for client, model in level1_clients:
        call = functools.partial(
            client.chat.completions.create, 
            model=model, 
            messages=messages, 
            **kwargs
        )
        tasks.append(asyncio.to_thread(call))

    level1_responses = await asyncio.gather(*tasks)

    print(level1_responses[0].choices[0].message.content)
    print(level1_responses[1].choices[0].message.content)

    strategy_class = STRATEGY_MAPPING.get(agreement_strategy)
    if not strategy_class:
        raise ValueError(f"Unknown agreement strategy: {agreement_strategy}")
    
    strategy = strategy_class()
    agreed = strategy.check_agreement(level1_responses)

    if agreed:
        print("Level 1 clients agreed. Returning first response.")
        return level1_responses[0]
    else:
        print("Level 1 clients disagreed. Escalating to Level 2 client.")
        
        l2_client, l2_model = level2_client

        call = functools.partial(
            l2_client.chat.completions.create,
            model=l2_model,
            messages=messages,
            **kwargs
        )
        
        level2_response = await asyncio.to_thread(call)
        
        print("Received response from Level 2 client.")
        return level2_response 