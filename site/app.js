(() => {
  "use strict";

  const root = document.documentElement;
  const readPreference = (key) => {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  };
  const writePreference = (key, value) => {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      // The portfolio remains fully usable when storage is unavailable.
    }
  };

  let language = readPreference("portfolio-language") === "ko" ? "ko" : "en";
  let activeFilter = "all";

  const filterButtons = [...document.querySelectorAll("[data-filter]")];
  const languageButtons = [...document.querySelectorAll("[data-language]")];
  const cards = [...document.querySelectorAll(".track-card")];
  const filterStatus = document.querySelector(".filter-status");
  const search = document.querySelector("#evidence-search");
  const ledgerRows = [...document.querySelectorAll("#evidence-ledger tbody tr")];
  const emptyState = document.querySelector(".empty-state");

  const updateFilterStatus = (visible) => {
    if (!filterStatus) return;
    filterStatus.textContent =
      language === "ko"
        ? visible + "개 / " + cards.length + "개 프로젝트"
        : visible + " of " + cards.length + " projects shown";
  };

  const applyCardFilter = () => {
    let visible = 0;
    cards.forEach((card) => {
      const classes = (card.dataset.evidence || "").split(/\s+/);
      const show = activeFilter === "all" || classes.includes(activeFilter);
      card.hidden = !show;
      if (show) visible += 1;
    });
    updateFilterStatus(visible);
  };

  const applyLedgerSearch = () => {
    if (!search) return;
    const query = search.value.trim().toLocaleLowerCase(language);
    let visible = 0;
    ledgerRows.forEach((row) => {
      const show = !query || row.textContent.toLocaleLowerCase(language).includes(query);
      row.hidden = !show;
      if (show) visible += 1;
    });
    if (emptyState) emptyState.hidden = visible !== 0;
  };

  const setLanguage = (nextLanguage) => {
    language = nextLanguage === "ko" ? "ko" : "en";
    root.lang = language;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const translated = node.dataset[language];
      if (translated) node.textContent = translated;
    });
    languageButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.language === language));
    });
    if (search) {
      search.placeholder = language === "ko" ? "evidence 검색…" : "Search evidence…";
      search.setAttribute(
        "aria-label",
        language === "ko" ? "evidence ledger 검색" : "Search evidence ledger",
      );
    }
    writePreference("portfolio-language", language);
    applyCardFilter();
    applyLedgerSearch();
  };

  languageButtons.forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.language));
  });

  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter || "all";
      filterButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      applyCardFilter();
    });
  });

  if (search) search.addEventListener("input", applyLedgerSearch);

  const themeButton = document.querySelector(".theme-toggle");
  const preferredTheme = readPreference("portfolio-theme");
  const systemPrefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
  let theme =
    preferredTheme === "light" || preferredTheme === "dark"
      ? preferredTheme
      : systemPrefersLight
        ? "light"
        : "dark";

  const setTheme = (nextTheme) => {
    theme = nextTheme === "light" ? "light" : "dark";
    root.dataset.theme = theme;
    const label =
      theme === "dark" ? "Switch to light color theme" : "Switch to dark color theme";
    if (themeButton) {
      themeButton.setAttribute("aria-label", label);
      themeButton.title = label;
    }
    writePreference("portfolio-theme", theme);
  };

  themeButton?.addEventListener("click", () => {
    setTheme(theme === "dark" ? "light" : "dark");
  });

  const pageSections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll(".primary-nav a")];
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        navLinks.forEach((link) => {
          const pointsToSection = link.hash === "#" + visible.target.id;
          if (pointsToSection) link.setAttribute("aria-current", "true");
          else link.removeAttribute("aria-current");
        });
      },
      { rootMargin: "-20% 0px -68%", threshold: [0, 0.2, 0.6] },
    );
    pageSections.forEach((section) => observer.observe(section));
  }

  const host = window.location.hostname;
  if (host.endsWith(".github.io")) {
    const owner = host.slice(0, -".github.io".length);
    const repository = window.location.pathname.split("/").filter(Boolean)[0];
    if (owner && repository) {
      document.querySelectorAll("[data-repo-path]").forEach((link) => {
        const path = link.dataset.repoPath;
        link.href =
          "https://github.com/" + owner + "/" + repository + "/blob/main/" + path;
        link.rel = "noopener";
      });
    }
  }

  setTheme(theme);
  setLanguage(language);
})();

