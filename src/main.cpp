#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <random>
#include "arial.h"

struct player
{
    double x, y, x_speed, y_speed;
};

struct inputs
{
    bool key_right, key_left, key_up, key_down, space;
};

const int MAX_NB_AIS = 100;
const double slipperyness = 0.95;
const int arena_size = 1000;
player players[MAX_NB_AIS + 1];

int screen_arena_size, x_arena, y_arena;

void update_player(int i_player, std::array<bool, 5> button_presse)
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
    players[i_player].x = 0;
    players[i_player].y = 0;
    players[i_player].x_speed = 0;
    players[i_player].y_speed = 0;
}

sf::CircleShape draw_player(int x, int y, sf::Color col)
{
    double scale = (double)screen_arena_size / arena_size;
    int real_x = x * scale + x_arena;
    int real_y = y * scale + y_arena;
    sf::CircleShape dot(screen_arena_size / 100.0);
    dot.setPosition(real_x, real_y);
    dot.setFillColor(col);
    return dot;
}

std::array<bool, 5> get_AI_input(int i_network)
{
    std::array<bool, 5> input;
    static std::default_random_engine generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int i = 0; i < 5; i++)
    {
        input[i] = (bool)distribution(generator);
    }
    return input;
}

int main()
{
    int window_width = 1000, window_height = 1000;
    int mode = 0;
    int nb_AIs = 50;
    x_arena = 500, y_arena = 500, screen_arena_size = 900;
    for (int i_player = 0; i_player < nb_AIs + 1; i_player++)
    {
        reset_player(i_player);
    }

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

                screen_arena_size = std::min(window_width - 100, window_height - 100);
                x_arena = window_width / 2, y_arena = window_height / 2;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        {
            mode = 0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
        {
            mode = 1;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
        {
            mode = 2;
        }

        if (mode == 0 || mode == 2)
        {
            std::array<bool, 5> inps = {
                sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::Q),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::Z),
                false};
            update_player(0, inps);
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                std::array<bool, 5> inps = get_AI_input(i_player);
                update_player(i_player, inps);
            }
        }

        window.clear(sf::Color::White);

        sf::RectangleShape arena(sf::Vector2f(screen_arena_size, screen_arena_size));
        arena.setPosition(x_arena - screen_arena_size / 2, y_arena - screen_arena_size / 2);
        arena.setFillColor(sf::Color(171, 203, 230));
        window.draw(arena);

        if (mode == 0 || mode == 2)
        {
            window.draw(draw_player(players[0].x, players[0].y, sf::Color(128, 0, 128)));
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                window.draw(draw_player(players[i_player].x, players[i_player].y, sf::Color(255, 0, 0)));
            }
        }

        window.display();
    }

    return 0;
}
