// Minimal example for a window.

#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <iostream>
int main() {
  graphic::GraphicContext context;
  context.run([&](graphic::GraphicContext *context, SDL_Window *) {
    {
      static int counter = 0; // A variable
      // Why static?
      //   https://github.com/ocornut/imgui/blob/master/imgui_demo.cpp#L23-L29

      ImGui::Begin("An ImGui Window"); // Window creation

      ImGui::Text("Static text in the window"); // A line of text

      if (ImGui::Button(
              "Increase counter")) { // A button, its label and its action
        counter++;
      }

      ImGui::SameLine();                    // annotation for the next element
      ImGui::Text("counter = %d", counter); // Text with formatted string

      ImGui::End(); // End
    }
  });
  return 0;
}
