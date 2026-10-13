//! egui overlay: a control panel mirroring the hotkeys, plus a key reference.

use crate::App;

/// Key reference shown in the panel, matching `App::on_key` and the movement keys.
const KEYS: &[(&str, &str)] = &[
    ("WASD / Q E", "move, down / up"),
    ("Right mouse", "look"),
    ("Up / Down", "c x1.25 / /1.25"),
    ("I", "infinite c"),
    ("F", "light on/off"),
    ("L", "flashlight / lamp"),
    ("M", "lamp circles / stays"),
    ("B", "ambient on/off"),
    ("T", "emission-time view"),
    ("V", "Doppler color on/off"),
    ("[ / ]", "Doppler strength"),
    ("R", "half / full resolution"),
    ("Space", "pause"),
    ("H", "hide / show this panel"),
    ("Esc", "quit"),
];

impl App {
    pub(crate) fn panel(&mut self, ctx: &egui::Context) {
        if !self.show_panel {
            return;
        }
        egui::Window::new("rustman")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Light speed");
                ui.add_enabled(
                    !self.infinite_c,
                    egui::Slider::new(&mut self.c, 1.0..=1000.0).logarithmic(true).suffix(" m/s").text("c"),
                );
                ui.checkbox(&mut self.infinite_c, "Infinite c (compare)");
                let c = self.effective_c();
                let speed = self.world.max_speed;
                let text = format!("fastest object: {speed:.1} m/s ({:.2}c)", speed / c);
                if speed > c {
                    ui.colored_label(ui.visuals().warn_fg_color, format!("{text}  faster than c!"));
                } else {
                    ui.label(text);
                }

                ui.separator();
                ui.heading("Light");
                ui.checkbox(&mut self.world.light.on, "On");
                let mut flashlight = self.world.light.flashlight;
                if ui.checkbox(&mut flashlight, "Flashlight (else ceiling lamp)").changed() {
                    self.toggle_flashlight();
                }
                ui.add_enabled(
                    !self.world.light.flashlight,
                    egui::Checkbox::new(&mut self.world.light.moving, "Lamp circles"),
                );
                ui.checkbox(&mut self.ambient, "Ambient fill");

                ui.separator();
                ui.heading("View");
                ui.checkbox(&mut self.debug_age, "Emission-time view");
                ui.checkbox(&mut self.doppler, "Doppler color");
                ui.add_enabled(
                    self.doppler,
                    egui::Slider::new(&mut self.doppler_strength, 0.02..=4.0)
                        .logarithmic(true)
                        .text("strength (1 = physical)"),
                );
                if let Some(gpu) = &mut self.gpu {
                    let mut half = gpu.render_scale < 1.0;
                    if ui.checkbox(&mut half, "Half resolution").changed() {
                        gpu.set_render_scale(if half { 0.5 } else { 1.0 });
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.paused, "Pause");
                    ui.label(format!("t = {:.1} s", self.world.time));
                });

                egui::CollapsingHeader::new("Keys").show(ui, |ui| {
                    egui::Grid::new("keys").striped(true).show(ui, |ui| {
                        for (key, action) in KEYS {
                            ui.monospace(*key);
                            ui.label(*action);
                            ui.end_row();
                        }
                    });
                });
            });
    }
}
