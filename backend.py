
"""""
This is a program that uses an API to generate a quiz based on a text about Christmas.

The script first sets the OpenAI API key and creates an empty list. It then prints a welcome message and enters a loop that continues until the user types "quit()".
Inside the loop, the script creates a message that contains the text about Christmas.This message is appended to the "messages" list.
The script then uses the API to generate a response to this message, which is stored in the "response" variable. 
The actual response text is extracted from the "response" variable and appended to the "messages" list with the role "assistant". Finally, the response is printed to the console.

"""

import openai

openai.api_key = "insert key here"

messages = []


print("Welcome to the Notes and QnA generator that you always dreamt of having. This is the quiz based on the text that you have provided")
while input != "quit()":
    message = "generate a quiz based on the text - Christmas is celebrated every year on December 25. The festival marks the celebration of the birth anniversary of Jesus Christ. Jesus Christ is worshipped as the Messiah of God in Christian Mythology. Hence, his birthday is one of the most joyous ceremonies amongst Christians. Although the festival is mainly celebrated by the followers of Christianity, it is one of the most enjoyed festivals all over the globe. Christmas symbolizes merriment and love. It is celebrated with a lot of zeal and enthusiasm by everyone no matter what religion they follow.The season of Christmas that begins from Thanksgiving brings festivity and joy to everyone’s lives. Thanksgiving is the day when people thank the almighty for blessing them with harvest and also show gratitude towards all the good things and people around. On Christmas, people wish each other Merry Christmas and pray that the day takes away all the negativity and darkness from people’s life. Christmas is a festival full of culture and tradition. The festival entails a lot of preparations. Preparations for Christmas start early for most people. Preparations for Christmas involve a lot of things including buying decorations, food items, and gifts for family members and friends. People usually wear white or red coloured outfits on the day of Christmas.The celebration begins with decorating a Christmas tree. Christmas tree decoration and lighting are the most important part of Christmas. The Christmas tree is an artificial or real pine tree that people adorn with lights, artificial stars, toys, bells, flowers, gifts, etc. People also hide gifts for their loved ones. Traditionally, gifts are hidden in socks under the tree. It is an old belief that a saint named Santa Claus comes on the night of Christmas eve and hides presents for well-behaved kids. This imaginary figure brings a smile to everyone’s face.Young children are especially excited about Christmas as they receive gifts and great Christmas treats. The treats include chocolates, cakes, cookies, etc. People on this day visit churches with their families and friends and light candles in front of the idol of Jesus Christ. Churches are decorated with fairy lights and candles. People also create fancy Christmas cribs and adorn them with gifts, lights, etc. Children sing Christmas carols and also perform various skits marking the celebration of the auspicious day. One of the famous Christmas carols sung by all is Jingle Bell, Jingle Bell, Jingle all the way.On this day, people tell each other stories and anecdotes related to Christmas. It is believed that Jesus Christ, the son of God, came to the Earth on this day to end people’s sufferings and miseries. His visit is symbolic of goodwill and happiness and it is depicted through the visit of the wise men and the shepherds. Christmas is, indeed, a magical festival that is all about sharing joy and happiness. For this reason, it is also my most favorite festival.Apart from the religious beliefs, the festival is known as sharing gifts with family as well as friends. The cute kids wait for the whole year to receive gifts from Santa. The craze of receiving gifts increases so much that they get up at midnight and start asking what they are going to get from Santa. They share their wishes with their parents and their parents try to accomplish them on the behalf of Santa"


    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")