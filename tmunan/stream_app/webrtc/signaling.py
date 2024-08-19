import os
import json
import uuid
import asyncio
from typing import Callable
from pydantic import BaseModel, Field

from ably import AblyRealtime

from tmunan.utils.log import get_logger


class Offer(BaseModel):
    id: uuid.UUID
    name: str
    output: bool
    sdp: str
    type: str


class Answer(BaseModel):
    id: uuid.UUID
    name: str
    sdp: str
    type: str


class AblySignalingChannel:

    def __init__(self):

        # env
        self.logger = get_logger(self.__class__.__name__)

        # ably client
        self.ably_api_key = os.environ.get('ABLY_API_KEY', None)
        self.ably_client = AblyRealtime(key=self.ably_api_key, auto_connect=False)
        self.ably_channel_name = None

        # callback
        self.on_offer: Callable[[Offer], Answer] | None = None

    def listen_to_offers(self, channel_name: str, on_offer: Callable[[Offer], Answer]):

        # save channel
        self.ably_channel_name = channel_name

        # save callback
        self.on_offer = on_offer

        # run async task to connect and subscribe to offers
        asyncio.create_task(self.connect())

    async def connect(self):

        # wait until connected state
        self.ably_client.connect()
        await self.ably_client.connection.once_async('connected')
        self.logger.info('Ably - Connected')

        # subscribe to "offer" messages on signaling channel
        await self.subscribe(
            channel=self.ably_channel_name,
            event_name='offer',
            on_message=self.on_signaling_message
        )

    async def on_signaling_message(self, message):

        # log
        self.logger.info(f"SignalingChannel - Incoming message {message.name=}")

        # check for offer (sanity)
        if message.name == 'offer':

            # peer id
            peer_id = uuid.uuid4()

            # load json from message
            offer_data = json.loads(message.data)

            # create offer model
            offer = Offer(id=peer_id, **offer_data)

            # handle offer
            answer = self.on_offer(offer)

            # publish answer
            await self.publish(
                channel=self.ably_channel_name,
                event_name='answer',
                message=json.dumps(answer.model_dump())
            )

    async def disconnect(self):

        # close connection
        await self.ably_client.close()

        # log
        self.logger.info('Ably - Disconnected')

    async def subscribe(self, channel: str, event_name: str, on_message):

        # get channel
        channel = self.ably_client.channels.get(channel)

        # subscribe
        await channel.subscribe(event_name, on_message)

        # log
        self.logger.info(f'Ably - Subscribed to {channel.name=} for {event_name=} messages')

    async def publish(self, channel: str, event_name: str, message: str):

        # get channel
        channel = self.ably_client.channels.get(channel)

        # publish message
        await channel.publish(event_name, message)

        # log
        self.logger.info(f'Ably - Published message to channel {channel.name=}, {event_name=}')
        self.logger.debug(f'Ably - Published message: {event_name=}, {message=}')
