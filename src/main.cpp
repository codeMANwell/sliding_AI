#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <random>
#include "arial.h"

#define MAX_NB_PLAYERS 100
#define MAX_NB_INPUTS 100
#define MAX_NB_HIDDEN_LAYERS 10
#define MAX_NB_PER_HIDDEN 100
#define MAX_NB_OUTPUTS 5

const int WeightArraySize = (MAX_NB_INPUTS * MAX_NB_PER_HIDDEN) +
                            MAX_NB_PER_HIDDEN * MAX_NB_PER_HIDDEN * (MAX_NB_HIDDEN_LAYERS - 1) +
                            MAX_NB_PER_HIDDEN * MAX_NB_OUTPUTS;
const int BiasesArraySize = MAX_NB_HIDDEN_LAYERS * MAX_NB_PER_HIDDEN + MAX_NB_OUTPUTS;

struct player
{
    double x, y, x_speed, y_speed;
};

struct inputs
{
    bool key_right, key_left, key_up, key_down, space;
};

static std::random_device rd;
static std::mt19937 generator(rd());
static std::uniform_real_distribution<double> distrib_d(0.0, 1.0);
static std::uniform_int_distribution<int> distrib_i(0, 1);

double get_rand_double01()
{
    return distrib_d(generator);
}

int get_rand_int01()
{
    return distrib_i(generator);
}

class SlidingGame
{
public:
    SlidingGame(double slipperyness_arg, double accel_arg, int arena_size_arg)
    {
        slipperyness = slipperyness_arg, accel = accel_arg, arena_size = arena_size_arg;
    }

    void update_player(int i_player, std::array<bool, 5> button_presse)
    {
        players[i_player].x_speed += (button_presse[0] ? accel : 0) - (button_presse[1] ? accel : 0);
        players[i_player].y_speed += (button_presse[2] ? accel : 0) - (button_presse[3] ? accel : 0);
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
    player get_player(int player_id)
    {
        return players[player_id];
    }
    int get_arena_size()
    {
        return arena_size;
    }

private:
    double slipperyness;
    double accel;
    int arena_size;
    player players[MAX_NB_PLAYERS + 1];
};

class NeuronalNetwork
{
public:
    NeuronalNetwork(int inp, int per_hid, int out, int nb_hid)
    {
        if (inp > MAX_NB_INPUTS || per_hid > MAX_NB_PER_HIDDEN ||
            nb_outputs > MAX_NB_OUTPUTS || nb_hid > MAX_NB_HIDDEN_LAYERS ||
            inp < 0 || per_hid < 0 || nb_outputs < 0 || nb_hid < 0)
        {
            throw std::invalid_argument("Invalid or out of range dimensions");
        }
        nb_inputs = inp, nb_per_hidden = per_hid, nb_outputs = out, nb_hidden_layers = nb_hid;
        randomize();
    }

    void randomize()
    {
        for (int i = 0; i < WeightArraySize + 1; i++)
        {
            weights[i] = 4 * get_rand_double01() - 2;
        }
    }

    std::vector<double> run_network(std::vector<double> inputs)
    {
        std::vector<double> res;
        for (int o = 0; o < nb_outputs; o++)
        {
            res.push_back(get_rand_double01());
        }
        return res;
    }

private:
    int nb_inputs, nb_per_hidden, nb_outputs, nb_hidden_layers;
    std::array<double, WeightArraySize> weights;
    std::array<double, BiasesArraySize> biases;

    int get_idx_w(int start_layer, int start_id, int end_id)
    {
        return 0;
    }
};

int screen_arena_size, x_arena, y_arena;

sf::CircleShape draw_player(int x, int y, int game_arena_size, sf::Color col)
{
    double scale = (double)screen_arena_size / game_arena_size;
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

    SlidingGame game(0.95, 1.0, 1000);

    std::vector<NeuronalNetwork> networks;
    for (int i = 0; i < nb_AIs; i++)
    {
        networks.push_back(NeuronalNetwork(4, 6, 5, 2));
    }

    for (int i_player = 0; i_player < nb_AIs + 1; i_player++)
    {
        game.reset_player(i_player);
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

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1))
        {
            mode = 0;
            // std::cout << get_rand_double01() << std::endl;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2))
        {
            mode = 1;
            // std::cout << get_rand_int01() << std::endl;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
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
            game.update_player(0, inps);
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                std::array<bool, 5> inps = get_AI_input(i_player);
                game.update_player(i_player, inps);
            }
        }

        window.clear(sf::Color::White);

        sf::RectangleShape arena(sf::Vector2f(screen_arena_size, screen_arena_size));
        arena.setPosition(x_arena - screen_arena_size / 2, y_arena - screen_arena_size / 2);
        arena.setFillColor(sf::Color(171, 203, 230));
        window.draw(arena);

        if (mode == 0 || mode == 2)
        {
            player p_drawn = game.get_player(0);
            window.draw(draw_player(p_drawn.x, p_drawn.y, game.get_arena_size(), sf::Color(128, 0, 128)));
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                player p_drawn = game.get_player(i_player);
                window.draw(draw_player(p_drawn.x, p_drawn.y, game.get_arena_size(), sf::Color(255, 0, 0)));
            }
        }

        window.display();
    }

    return 0;
}
