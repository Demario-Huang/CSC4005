// Do without GUI.

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

int main(int argc, char **argv) {
  graphic::GraphicContext context;

  // Read parameters
  bool sort_immediately = false;
  if (argc == 2 && argv[1][0] == 'y') {
    sort_immediately = true;
    printf("Starting sort immediately\n");
  }

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
      if (clicked || sort_immediately) {
        if (!already) {
          already = bubble_sort(arr);
          printf("Bubble sorted: ");
          for (uint i = 0; i < arr.size(); i++) {
            printf("%d ", arr[i]);
          }
          printf("\n");
        } else {
          printf("Sort done! Bye!");
          exit(0);
        }
      }

      ImGui::SliderInt("Display", &display_count, 0, 9);

      ImGui::Text("Already sorted: %d\n", already);

      ImGui::Text("Numbers:");

      for (uint i = 0; i < display_count; i++) {
        ImGui::SameLine();
        ImGui::Text("%d ", arr[i]);
      }

      ImGui::End();
    }
  });
  return 0;
}