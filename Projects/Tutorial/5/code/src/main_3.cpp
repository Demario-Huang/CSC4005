// Interactive actions and result display.

#include <algorithm>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <iostream>
#include <vector>

bool bubble_sort(std::vector<int> &vec) {
  bool already_sorted = true;
  for (uint i = 0; i < vec.size() - 1; i++) {
    if (vec[i + 1] < vec[i]) {
      int m = vec[i];
      vec[i] = vec[i + 1];
      vec[i + 1] = m;
      already_sorted = false;
    }
  }
  return already_sorted;
}

int main() {
  graphic::GraphicContext context;

  // Our variables
  std::vector<int> arr = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  bool already = false;
  int display_count = 9;

  // Our window
  context.run([&](graphic::GraphicContext *context, SDL_Window *) {
    {
      ImGui::Begin("Click to sort");

      ImGui::Text("Sort:");

      ImGui::SameLine();
      bool clicked = ImGui::Button("Run");
      // Sample 1: respond to click event
      if (clicked) {
        // Sample 2: modify variables outside GUI
        already = bubble_sort(arr);
        printf("Bubble sorted: ");
        for (uint i = 0; i < arr.size(); i++) {
          printf("%d ", arr[i]);
        }
        printf("\n");
      }

      // Sample 3: bind input from GUI to outside variables
      ImGui::SliderInt("Display", &display_count, 0, 9);

      // Sample 4: a widget that reads outside variable
      ImGui::Text("Already sorted: %d\n", already);

      ImGui::Text("Numbers:");

      // Sample 4: a widget that reads outside variable
      for (uint i = 0; i < display_count; i++) {
        ImGui::SameLine();
        ImGui::Text("%d ", arr[i]);
      }

      ImGui::End();
    }
  });
  return 0;
}