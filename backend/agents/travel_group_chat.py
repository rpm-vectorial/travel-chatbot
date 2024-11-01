import asyncio
from collections import defaultdict
from typing import List

from autogen_core.base import AgentId, MessageContext
from autogen_core.components import (DefaultTopicId, RoutedAgent,
                                     message_handler, type_subscription)

from ..data_types import (AgentResponse, EndUserMessage, GroupChatMessage,
                          TravelPlan, TravelRequest)
from ..otlp_tracing import logger


@type_subscription("group_chat_manager")
class GroupChatManager(RoutedAgent):
    """
    Manages communication between multiple agents involved in creating a travel plan.

    Attributes:
        _chat_history (List[GroupChatMessage]): Stores messages exchanged during the chat.
        _conversation_complete (bool): Indicates if the conversation is complete.
        _session_id (str): Stores the current session ID.
        _responses (defaultdict): Stores agent responses for compiling the final travel plan.
    """

    def __init__(self) -> None:
        super().__init__("GroupChatManager")
        self._chat_history: List[GroupChatMessage] = []
        self._conversation_complete = False
        self._session_id = None
        self._responses = defaultdict(list)

    @message_handler
    async def handle_travel_request(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        """
        Handles incoming travel requests from the user and initiates communication with relevant agents.

        Args:
            message (EndUserMessage): The incoming user message.
            ctx (MessageContext): The context of the current message.
        """
        logger.info(f"GroupChatManager received travel request: {message.content}")
        self._session_id = ctx.topic_id.source
        await self.request_relevant_agents(ctx.topic_id.type)

    @message_handler
    async def handle_complex_travel_request(
        self, message: TravelPlan, ctx: MessageContext
    ) -> None:
        """
        Handles complex travel requests that require coordination between multiple agents.

        Args:
            message (TravelPlan): The incoming travel plan request containing multiple tasks.
            ctx (MessageContext): The context of the current message.
        """
        logger.info(f"GroupChatManager received complex travel request: {message}")
        self._session_id = ctx.topic_id.source
        tasks = [
            self.send_message(
                TravelRequest(
                    source="GroupChatManager",
                    content=task.task_details,
                    original_task=message.main_task,
                ),
                AgentId(type=task.assigned_agent, key=self._session_id),
            )
            for task in message.subtasks
        ]
        group_results: List[GroupChatMessage] = await asyncio.gather(*tasks)

        logger.info(f"GroupChatManager received responses from agents: {group_results}")
        # Compile the final travel plan based on agent responses
        final_plan = "\n".join([response.content for response in group_results])
        await self.publish_message(
            AgentResponse(
                source="GroupChatManager",
                content=f"Here is your comprehensive travel plan:\n{final_plan}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    async def request_relevant_agents(self, relevant_agents: List[str]) -> None:
        """
        Sends requests to the relevant agents to provide details needed for the travel plan.

        Args:
            relevant_agents (List[str]): The list of agent types involved in the travel plan.
        """
        logger.info(
            "GroupChatManager requesting relevant agents to provide details for the travel plan"
        )
        for agent_type in relevant_agents:
            await self.publish_message(
                TravelRequest(
                    source="GroupChatManager",
                    content="Provide details for the travel plan",
                    requirements={"destination_city": "Paris"},
                ),
                DefaultTopicId(type=agent_type, source=self._session_id),
            )

    @message_handler
    async def handle_handoff(self, message: TravelRequest, ctx: MessageContext) -> None:
        """
        Handles handoff requests from other agents, either continuing or concluding the conversation.

        Args:
            message (TravelRequest): The handoff message from another agent.
            ctx (MessageContext): Context information for the message.
        """
        session_id = ctx.topic_id.source
        logger.info(f"Received handoff message from {message.source}")

        if message.complete:
            self._conversation_complete = True
            logger.info("Conversation completed. Clearing session.")
            # Add cleanup or finalization logic here if needed.
        else:
            await self.compile_final_plan()
