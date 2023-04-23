import openai

openai.api_key = "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"

messages = []


text = "Dogs are one of the most popular and beloved pets around the world. They come in various shapes, sizes, and breeds, and have been domesticated for thousands of years. Dogs are highly social animals and thrive on companionship with humans and other dogs. They are known for their loyalty, intelligence, and ability to form strong bonds with their owners. Dogs can be trained to perform a wide range of tasks, from assisting people with disabilities to serving as therapy animals. They also provide a great source of comfort and companionship to their owners and are known to reduce stress and promote relaxation. With their unique personalities and unwavering devotion, it's no wonder that dogs are often referred to as man's best friend"
while input != "quit()":
    message = "generate a quiz based on the text"
    message = message + text


    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")