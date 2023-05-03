<p class="py-1 tracking-wide leading-relaxed {$$props.class}">
  <slot/>
</p>

<style>
  .conversation{
    white-space: pre-wrap;
  }

@media only screen and (min-width: 768px) {
  .conversation{
    padding-left: 1.75rem;
    padding-right: 1.75rem;
  }
}

@media only screen and (max-width: 767px) {
  .conversation{
    margin-top: 1rem;
    margin-bottom: 1rem;
    /* padding: 0 0.3rem; */
    border-style: solid;
    border-width: 0px;
    border-image: linear-gradient(to right, var(--background), var(--text), var(--text), var(--text), var(--background)) 1% 0%;
    background-position: 50% 100%;
    background-repeat: no-repeat;
  }
}
</style>