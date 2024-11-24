import random
from typing import Dict, List
import os
import json
from autogen_core.base import MessageContext
from autogen_core.components import (
    DefaultTopicId,
    RoutedAgent,
    message_handler,
    type_subscription,
)
from typing_extensions import Annotated
from autogen_core.components.tools import FunctionTool, Tool

from ..data_types import (
    AgentResponse,
    EndUserMessage,
    GroupChatMessage,
    HandoffMessage,
    TravelRequest,
    FlightBooking,
    AgentStructuredResponse,
)

from openai import AzureOpenAI
from ..otlp_tracing import logger


async def simulate_flight_booking(
    flight_params: dict
) -> FlightBooking:

    departure_city: str = flight_params["departure_city"]
    destination_city: str = flight_params["destination_city"]
    departure_date: str = flight_params["departure_date"]
    return_date: str = flight_params["return_date"]
    number_of_passengers: int = flight_params["number_of_passengers"]
    
    flight_options = [
        {"airline": "Air France", "flight_number": "AF123", "price_per_ticket": 200},
        {"airline": "Delta", "flight_number": "DL456", "price_per_ticket": 250},
        {
            "airline": "British Airways",
            "flight_number": "BA789",
            "price_per_ticket": 300,
        },
        {"airline": "Lufthansa", "flight_number": "LH101", "price_per_ticket": 220},
        {"airline": "Emirates", "flight_number": "EK202", "price_per_ticket": 400},
    ]

    selected_flight = random.choice(flight_options)
    total_price = 2 * selected_flight["price_per_ticket"]
    booking_reference = (
        f"FL-{random.randint(1000, 9999)}-{destination_city[:3].upper()}"
    )

    return FlightBooking(
        departure_city=departure_city,
        destination_city=destination_city,
        departure_date=departure_date,
        return_date=return_date,
        airline=selected_flight["airline"],
        flight_number=selected_flight["flight_number"],
        total_price=total_price,
        booking_reference=booking_reference,
        number_of_passengers=number_of_passengers,
    )


def get_flight_booking_tool() -> List[Tool]:
    return [
        FunctionTool(
            name="simulate_flight_booking",
            func=simulate_flight_booking,
            description="This function simulates the process of booking a flight. It takes the departure city, destination city, departure date, and return date as input and returns the flight booking details, including the departure and destination cities, departure and return dates, airline, flight number, total price, and booking reference. This function is useful when the user wants to book a flight.",
        )
    ]


# This agent is responsible for handling flight booking requests from the user.
# It simulates a flight booking process and returns a response with the booking details.
# NOTE: There is no LLM in this agent and we are simulating the flight booking process.
@type_subscription(topic_type="flight_booking")
class FlightAgent(RoutedAgent):

    _details_extractor = f"""You are simply supposed to extract the proper information from\
                             the message provided to you as context.
                            
                             Example: 
                             1. message: i want to book a flight from New York to paris from 2023-12-20 to 2023-12-30.
                                response:     <departure_city : "New York",
                                              destination_city : "Paris",
                                              departure_date : "2023-12-20",
                                              return_date : "2023-12-30",
                                              number_of_passengers : 2>
                             Note:
                             1. If no start or end dates are mentioned in the query, take the dates mentioned in the example.
                             2. Give your response strictly as a python dictionary with the keys mentioned in the example above.
                             3. if the number of passesngers is not mentioned in the qquery, take the default as 2.
                           """
    
    def __init__(self) -> None:
        super().__init__("FlightAgent")
        self._openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        )

    def _get_flight_details(self, message: str):
        try:
            logger.info(f"Extracting flight params")
            response = self._openai_client.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role" : "system",
                        "content" : FlightAgent._details_extractor
                    },
                    {
                        "role" : "user",
                        "content" : message
                    }
                ]
            )
        
            response = json.loads(response.choices[0].message.content.replace("\n", "").replace(" ", "").replace("python", "").replace("`", ""))
            logger.info(f"The extracted flight params are : {response}")
            return response
        
        except:
            logger.error("Error retrieving flight params")

        
    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        logger.info(f"FlightAgent received message: {message.content}")
        if "travel plan" in message.content.lower():
            await self.publish_message(
                HandoffMessage(content=message.content, source=self.id.type),
                DefaultTopicId(type="router", source=ctx.topic_id.source),
            )
            return

    
        flight_params = self._get_flight_details(message.content)
        
        response = await simulate_flight_booking(flight_params)
        
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=response,
                message=f"Simulated response: Flight booking processed successfully for query - {message.content}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        logger.info(f"FlightAgent received travel request sub-task: {message.content}")

        response = await simulate_flight_booking()
        return GroupChatMessage(
            source=self.id.type,
            content=f"Flight booking processed: {response}",
        )
