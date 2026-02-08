#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "arial.h"

struct player
{
    double x, y, x_speed, y_speed;
};

const int MAX_NB_AIS = 100;
const double slipperyness = 0.95;
player players[MAX_NB_AIS + 1];

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
    players[0].x = 500;
    players[0].y = 500;
    players[0].x_speed = 0;
    players[0].y_speed = 0;
}

int main()
{
    int window_width = 1000, window_height = 1000;
    int x_arena = 500, y_arena = 500, arena_size = 1000;
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
        sf::VideoMode(window_width, window_height),
        "Sliding-AI");

    window.setFramerateLimit(60);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
            if (event.type == sf::Event::Resized)
            {
                window_width = event.size.width;
                window_height = event.size.height;

                sf::FloatRect visibleArea(0, 0, window_width, window_height);
                window.setView(sf::View(visibleArea));

                arena_size = std::min(window_width, window_height);
                x_arena = window_width / 2, y_arena = window_height / 2;
            }
        }
        if (mode == 0 || mode == 2)
        {
            bool inputs[5] = {
                sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::Q),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::Z),
                false};
            update_player(0, inputs);
        }
        window.clear(sf::Color::White);

        sf::RectangleShape arena(sf::Vector2f(arena_size, arena_size));
        arena.setPosition(x_arena, y_arena);
        arena.setFillColor(sf::Color(128, 128, 128));
        window.draw(arena);

        if (mode == 0)
        {
            sf::CircleShape dot(10);
            dot.setPosition(players[0].x, players[0].y);
            dot.setFillColor(sf::Color(255, 0, 0));
            window.draw(dot);
        }

        window.display();
    }

    return 0;
}
