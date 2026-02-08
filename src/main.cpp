#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "arial.h"

const int MAX_NB_AIS = 100;
const float slipperyness = 0.9;
player players[MAX_NB_AIS + 1];

struct player
{
    int x, y, x_speed, y_speed;
};

void update_player(int i_player, bool button_presse[5])
{
    players[i_player].x_speed += (button_presse[0] ? 1 : 0) - (button_presse[1] ? 1 : 0);
    players[i_player].y_speed += (button_presse[2] ? 1 : 0) - (button_presse[3] ? 1 : 0);
    players[i_player].x_speed *= slipperyness;
    players[i_player].y_speed *= slipperyness;
    players[i_player].x += players[i_player].x_speed;
    players[i_player].y += players[i_player].y_speed;
}

void reset_player(int i_player)
{
    players[0].x = 50;
    players[0].y = 50;
    players[0].x_speed = 0;
    players[0].y_speed = 0;
}

int main()
{
    int windowWidth = 800, windowHeight = 600;
    int mode = 0;
    int nb_AIs = 50;
    reset_player(0);

    sf::Font font;
    if (!font.loadFromMemory(arial_ttf, arial_ttf_len))
    {
        std::cerr << "Text error" << std::endl;
        return -1;
    }

    sf::RenderWindow window(
        sf::VideoMode(800, 600),
        "Sliding-AI");

    window.setFramerateLimit(240);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        if (mode == 0 || mode == 2)
        {
            bool inputs[5] = {sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::Z),
                              sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S),
                              sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::Q),
                              sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D),
                              false};
            update_player(0, inputs);
        }
        window.clear(sf::Color::White);

        sf::CircleShape clearbutton(10);
        clearbutton.setPosition(100, 100);
        clearbutton.setFillColor(sf::Color(255, 0, 0));
        window.draw(clearbutton);

        window.display();
    }

    return 0;
}
