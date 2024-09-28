import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from neural_network import NeuralNetwork
from pokemon_dataset import PokemonDataset
import os
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def calculate_accuracy(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class indices

            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = (correct / total) * 100  # Convert to percentage
    return accuracy

def setup(): 
    # Load the dataset from Hugging Face
    ds = load_dataset("keremberke/pokemon-classification", name="full")

    # Transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Batch size
    batch_size = 4

    train_dataset = PokemonDataset(ds['train'], transform=transform)
    test_dataset = PokemonDataset(ds['test'], transform=transform)

    # Creating loaders for the training and test data
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Class names
    classes = ('Porygon', 'Goldeen', 'Hitmonlee', 'Hitmonchan', 'Gloom', 'Aerodactyl', 'Mankey', 'Seadra', 
               'Gengar', 'Venonat', 'Articuno', 'Seaking', 'Dugtrio', 'Machop', 'Jynx', 'Oddish', 
               'Dodrio', 'Dragonair', 'Weedle', 'Golduck', 'Flareon', 'Krabby', 'Parasect', 'Ninetales', 
               'Nidoqueen', 'Kabutops', 'Drowzee', 'Caterpie', 'Jigglypuff', 'Machamp', 'Clefairy', 
               'Kangaskhan', 'Dragonite', 'Weepinbell', 'Fearow', 'Bellsprout', 'Grimer', 'Nidorina', 
               'Staryu', 'Horsea', 'Electabuzz', 'Dratini', 'Machoke', 'Magnemite', 'Squirtle', 
               'Gyarados', 'Pidgeot', 'Bulbasaur', 'Nidoking', 'Golem', 'Dewgong', 'Moltres', 'Zapdos', 
               'Poliwrath', 'Vulpix', 'Beedrill', 'Charmander', 'Abra', 'Zubat', 'Golbat', 'Wigglytuff', 
               'Charizard', 'Slowpoke', 'Poliwag', 'Tentacruel', 'Rhyhorn', 'Onix', 'Butterfree', 
               'Exeggcute', 'Sandslash', 'Pinsir', 'Rattata', 'Growlithe', 'Haunter', 'Pidgey', 'Ditto', 
               'Farfetchd', 'Pikachu', 'Raticate', 'Wartortle', 'Vaporeon', 'Cloyster', 'Hypno', 'Arbok', 
               'Metapod', 'Tangela', 'Kingler', 'Exeggutor', 'Kadabra', 'Seel', 'Voltorb', 'Chansey', 
               'Venomoth', 'Ponyta', 'Vileplume', 'Koffing', 'Blastoise', 'Tentacool', 'Lickitung', 
               'Paras', 'Clefable', 'Cubone', 'Marowak', 'Nidorino', 'Jolteon', 'Muk', 'Magikarp', 
               'Slowbro', 'Tauros', 'Kabuto', 'Spearow', 'Sandshrew', 'Eevee', 'Kakuna', 'Omastar', 
               'Ekans', 'Geodude', 'Magmar', 'Snorlax', 'Meowth', 'Pidgeotto', 'Venusaur', 'Persian', 
               'Rhydon', 'Starmie', 'Charmeleon', 'Lapras', 'Alakazam', 'Graveler', 'Psyduck', 'Rapidash', 
               'Doduo', 'Magneton', 'Arcanine', 'Electrode', 'Omanyte', 'Poliwhirl', 'Mew', 
               'Alolan Sandslash', 'Mewtwo', 'Weezing', 'Gastly', 'Victreebel', 'Ivysaur', 'MrMime', 
               'Shellder', 'Scyther', 'Diglett', 'Primeape', 'Raichu')
    
    neural = NeuralNetwork()

    if os.path.exists('./model/pkmn_net.pth'):
        neural.load_state_dict(torch.load('./model/pkmn_net.pth'))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(neural.parameters(), lr=0.001, momentum=0.9)

        print('Starting Training')
        print('=================')
    
        for epoch in range(20):  # Loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainLoader):
                print(i)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = neural(inputs)  # Use the correct variable name for your model
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # Print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        PATH = './model/pkmn_net.pth'
        torch.save(neural.state_dict(), PATH)

    # dataiter = iter(testLoader)
    # next(dataiter)
    
    # next(dataiter)
    # next(dataiter)
    # next(dataiter)
    # next(dataiter)
    # next(dataiter)
    # next(dataiter)
    
    # next(dataiter)
    # images, labels = next(dataiter)

    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # outputs = neural(images)
    # _, predicted = torch.max(outputs, 1)


    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                             for j in range(4)))
    neural.eval()  # Mettre le modèle en mode évaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural.to(device)
    
    accuracy = calculate_accuracy(neural, testLoader, device)
    print(f'Accuracy on the test set: {accuracy:.2f}%')  # Print as percentage

if __name__ == '__main__':
    setup()
