module.exports = {
  version: "5.0",
  title: "ZastTranslate — Beta 1.18",
  description: "Video translation & dubbing with voice cloning — 100% local, zero API. Supports 30 languages (VoxCPM 2), YouTube SEO Studio, Viral Shorts Studio (9:16), and WordPress SEO Blog Post Generator.",
  icon: "zastttranslate.png",
  menu: async (kernel, info) => {
    let installed = info.exists("env/installed.sentinel")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      flux_install: info.running("flux_install.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (running.flux_install) {
      return [{
        default: true,
        icon: "fa-solid fa-wand-magic-sparkles",
        text: "Installing FLUX",
        href: "flux_install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js"
        }, {
          icon: "fa-solid fa-wand-magic-sparkles",
          text: "Install FLUX (Optional)",
          href: "flux_install.js"
        }, {
          icon: "fa-solid fa-sync",
          text: "Update",
          href: "update.js"
        }, {
          icon: "fa-solid fa-plug",
          text: "Reinstall",
          href: "install.js"
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Reset",
          href: "reset.js"
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }]
    }
  }
}
