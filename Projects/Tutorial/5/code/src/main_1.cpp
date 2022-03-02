// Reference file to find features you may need.
// Check the demo window for even more examples:
//   https://github.com/ocornut/imgui/blob/master/imgui_demo.cpp

#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <iostream>
int main() {
  graphic::GraphicContext context;
  bool show_demo_window = true;
  bool show_another_window = false;
  context.run([&](graphic::GraphicContext *context, SDL_Window *) {
    if (show_demo_window)
      ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair
    // to created a named window.
    {
      static float f = 0.0f;
      static int counter = 0;

      ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!"
                                     // and append into it.

      ImGui::Text("This is some useful text."); // Display some text (you can
                                                // use a format strings too)
      ImGui::Checkbox(
          "Demo Window",
          &show_demo_window); // Edit bools storing our window open/close state
      ImGui::Checkbox("Another Window", &show_another_window);

      ImGui::SliderFloat("float", &f, 0.0f,
                         1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
      ImGui::ColorEdit3(
          "clear color",
          (float *)&context
              ->clear_color()); // Edit 3 floats representing a color

      if (ImGui::Button("Button")) // Buttons return true when clicked (most
                                   // widgets return true when edited/activated)
        counter++;
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }

    // 3. Show another simple window.
    if (show_another_window) {
      ImGui::Begin(
          "Another Window",
          &show_another_window); // Pass a pointer to our bool variable (the
                                 // window will have a closing button that will
                                 // clear the bool when clicked)
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      ImGui::Text("Primitives");
      static float sz = 36.0f;
      static float thickness = 4.0f;
      static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
      ImGui::DragFloat("Size", &sz, 0.2f, 2.0f, 72.0f, "%.0f");
      ImGui::DragFloat("Thickness", &thickness, 0.05f, 1.0f, 8.0f, "%.02f");
      ImGui::ColorEdit4("Color", &col.x);
      {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImU32 col32 = ImColor(col);
        float x = p.x + 4.0f, y = p.y + 4.0f, spacing = 8.0f;
        for (int n = 0; n < 2; n++) {
          // First line uses a thickness of 1.0, second line uses the
          // configurable thickness
          float th = (n == 0) ? 1.0f : thickness;
          draw_list->AddCircle(ImVec2(x + sz * 0.5f, y + sz * 0.5f), sz * 0.5f,
                               col32, 6, th);
          x += sz + spacing; // Hexagon
          draw_list->AddCircle(ImVec2(x + sz * 0.5f, y + sz * 0.5f), sz * 0.5f,
                               col32, 20, th);
          x += sz + spacing; // Circle
          draw_list->AddRect(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 0.0f,
                             ImDrawCornerFlags_All, th);
          x += sz + spacing;
          draw_list->AddRect(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 10.0f,
                             ImDrawCornerFlags_All, th);
          x += sz + spacing;
          draw_list->AddRect(
              ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 10.0f,
              ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_BotRight, th);
          x += sz + spacing;
          draw_list->AddTriangle(ImVec2(x + sz * 0.5f, y),
                                 ImVec2(x + sz, y + sz - 0.5f),
                                 ImVec2(x, y + sz - 0.5f), col32, th);
          x += sz + spacing;
          draw_list->AddLine(ImVec2(x, y), ImVec2(x + sz, y), col32, th);
          x += sz + spacing; // Horizontal line (note: drawing a filled
                             // rectangle will be faster!)
          draw_list->AddLine(ImVec2(x, y), ImVec2(x, y + sz), col32, th);
          x += spacing; // Vertical line (note: drawing a filled rectangle will
                        // be faster!)
          draw_list->AddLine(ImVec2(x, y), ImVec2(x + sz, y + sz), col32, th);
          x += sz + spacing; // Diagonal line
          draw_list->AddBezierCurve(
              ImVec2(x, y), ImVec2(x + sz * 1.3f, y + sz * 0.3f),
              ImVec2(x + sz - sz * 1.3f, y + sz - sz * 0.3f),
              ImVec2(x + sz, y + sz), col32, th);
          x = p.x + 4;
          y += sz + spacing;
        }
        draw_list->AddCircleFilled(ImVec2(x + sz * 0.5f, y + sz * 0.5f),
                                   sz * 0.5f, col32, 6);
        x += sz + spacing; // Hexagon
        draw_list->AddCircleFilled(ImVec2(x + sz * 0.5f, y + sz * 0.5f),
                                   sz * 0.5f, col32, 32);
        x += sz + spacing; // Circle
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + sz, y + sz), col32);
        x += sz + spacing;
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + sz, y + sz), col32,
                                 10.0f);
        x += sz + spacing;
        draw_list->AddRectFilled(
            ImVec2(x, y), ImVec2(x + sz, y + sz), col32, 10.0f,
            ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_BotRight);
        x += sz + spacing;
        draw_list->AddTriangleFilled(ImVec2(x + sz * 0.5f, y),
                                     ImVec2(x + sz, y + sz - 0.5f),
                                     ImVec2(x, y + sz - 0.5f), col32);
        x += sz + spacing;
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + sz, y + thickness),
                                 col32);
        x += sz + spacing; // Horizontal line (faster than AddLine, but only
                           // handle integer thickness)
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + thickness, y + sz),
                                 col32);
        x += spacing + spacing; // Vertical line (faster than AddLine, but only
                                // handle integer thickness)
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + 1, y + 1), col32);
        x += sz; // Pixel (faster than AddLine)
        draw_list->AddRectFilledMultiColor(
            ImVec2(x, y), ImVec2(x + sz, y + sz), IM_COL32(0, 0, 0, 255),
            IM_COL32(255, 0, 0, 255), IM_COL32(255, 255, 0, 255),
            IM_COL32(0, 255, 0, 255));
        ImGui::Dummy(ImVec2((sz + spacing) * 8, (sz + spacing) * 3));
      }
      ImGui::End();
    }
  });
  return 0;
}